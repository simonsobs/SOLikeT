import json
import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
import sacc
from cobaya.functions import chi_squared


def _to_hashable(obj):
    """Recursively turn JSON-decoded lists back into (hashable) tuples."""
    if isinstance(obj, list):
        return tuple(_to_hashable(x) for x in obj)
    return obj


class GaussianData:
    """Container for named multivariate Gaussian data.

    Stores a data vector with its covariance matrix and provides methods
    for computing the Gaussian log-likelihood.

    Parameters
    ----------
    name : str
        Name identifier for the data
    x : Sequence
        Labels or coordinates for each data point (e.g., ell values)
    y : Sequence[float]
        The data vector values
    cov : np.ndarray
        Covariance matrix with shape (n, n) where n = len(x)
    ncovsims : int, optional
        Number of simulations used to estimate covariance. If provided,
        applies the Hartlap correction factor to the inverse covariance.
    indices : np.ndarray, optional
        Boolean array for trimming cross-covariances when scale cuts are applied
    ids : sequence, optional
        Per-bandpower identity keys over the FULL (pre-cut) range, i.e. one key
        per element of ``indices``. Lets ``MultiGaussianData`` align cross-
        covariance blocks to the data by identity instead of by position.

    Attributes
    ----------
    inv_cov : np.ndarray
        Inverse covariance matrix (with Hartlap correction if applicable)
    norm_const : float
        Normalization constant for the Gaussian likelihood

    Raises
    ------
    ValueError
        If dimensions of x, y, and cov are incompatible
        If covariance matrix has non-positive determinant
    """

    name: str  # name identifier for the data
    x: Sequence  # labels for each data point
    y: np.ndarray  # data point values
    cov: np.ndarray  # covariance matrix
    inv_cov: np.ndarray  # inverse covariance matrix
    ncovsims: int | None  # number of simulations used to estimate covariance
    indices: np.ndarray | None  # boolean array to trim cross-cov with selected bandpowers
    ids: list | None  # per-bandpower identity over the full (pre-cut) range

    _fast_chi_squared = staticmethod(chi_squared)

    def __init__(
        self,
        name,
        x: Sequence,
        y: Sequence[float],
        cov: np.ndarray,
        ncovsims: int | None = None,
        indices: np.ndarray | None = None,
        ids: Sequence | None = None,
    ):
        self.name = str(name)
        self.ncovsims = ncovsims
        self.indices = (
            indices
            if indices is not None and not all(indices)
            else np.ones(len(x), dtype=bool)
        )
        # Per-bandpower identity keys over the FULL (pre-cut) range, letting
        # ``MultiGaussianData`` align cross-covariance blocks by identity instead
        # of by position. ``None`` falls back to positional alignment.
        self.ids = list(ids) if ids is not None else None
        if self.ids is not None and len(self.ids) != len(self.indices):
            raise ValueError(
                f"ids has length {len(self.ids)}, expected one key per element of "
                f"the full (pre-cut) range len(indices)={len(self.indices)}."
            )

        if not (len(x) == len(y) and cov.shape == (len(x), len(x))):
            raise ValueError(
                f"Incompatible shapes! x={len(x)}, y={len(y)}, \
                               cov={cov.shape}"
            )

        self.x: Sequence[float] = x
        self.y: np.ndarray = np.ascontiguousarray(y)
        self.cov: np.ndarray = cov
        # self.eigenevalues = np.linalg.eigvalsh(cov)
        # if self.eigenevalues.min() <= 0:
        #    print(self.eigenevalues)
        #    raise ValueError("Covariance is not positive definite!")

        self.inv_cov: np.ndarray = np.linalg.inv(self.cov)
        if ncovsims is not None:
            hartlap_factor = (self.ncovsims - len(x) - 2) / (self.ncovsims - 1)
            self.inv_cov *= hartlap_factor
        # log_det = np.log(self.eigenevalues).sum()
        sign_log_det, log_det = np.linalg.slogdet(self.cov)
        if sign_log_det != 1:
            raise ValueError(
                f"Negative or zero determinant: \
                               sign(det)={sign_log_det}"
            )
        self.norm_const = -(np.log(2 * np.pi) * len(x) + log_det) / 2

    def __len__(self) -> int:
        return len(self.x)

    def loglike(self, theory: np.ndarray) -> float:
        """Compute the Gaussian log-likelihood.

        Parameters
        ----------
        theory : np.ndarray
            Theory prediction vector with same length as data

        Returns
        -------
        float
            Log-likelihood value including normalization constant
        """
        delta = self.y - theory
        return -0.5 * self._fast_chi_squared(self.inv_cov, delta) + self.norm_const


class CrossCov(dict):
    """Labelled-block covariance store for multi-component Gaussian likelihoods.

    A ``CrossCov`` is a labelled-block store: a dict whose keys are pairs of
    component names (e.g. ``("mflike", "lensing")``) and whose values are the
    corresponding covariance blocks. Diagonal keys ``(name, name)`` hold a
    component's auto-covariance; off-diagonal keys ``(name1, name2)`` hold a
    cross-covariance. ``add_component`` and ``add_cross_covariance`` are the same
    underlying store operation — they only differ in whether the block lands on
    the diagonal or off it.

    Each block may carry per-axis bandpower identities (``ids``), recorded in
    ``_block_ids_map`` keyed by the block's ``(row, col)`` tuple. These labels let
    a block be aligned to a target order regardless of how it was stored: blocks
    may be supplied in any order, on a full (un-cut, shuffled) range, with no
    reliance on positional or borrowed ids. ``MultiGaussianData`` canonicalises
    the store to the data order at assembly time via :meth:`to_canonical`, which
    fuses realignment and scale-cut trimming into one identity gather per axis.
    Supports saving and loading in SACC format for persistence.

    See ``.claude/plans/2026-06-05-crosscov-labelled-blocks-design.md`` for the
    design rationale.

    Examples
    --------
    Auto blocks via ``add_component`` and cross blocks via
    ``add_cross_covariance``, optionally labelled with per-axis ids::

        cross_cov = CrossCov()
        cross_cov.add_component("mflike", mflike_cov, ids=mflike_ids)
        cross_cov.add_component("lensing", lensing_cov, ids=lensing_ids)
        cross_cov.add_cross_covariance(
            "mflike", "lensing", cross_block, ids1=mflike_ids, ids2=lensing_ids
        )
        cross_cov.save("cross_cov.fits")

    Auto-covariances may be omitted; assembly then falls back to each
    likelihood's own ``cov``::

        cross_cov = CrossCov()
        cross_cov.add_cross_covariance("mflike", "lensing", cross_block)
        cross_cov.save("cross_cov.fits")

    **Loading**::

        cross_cov = CrossCov.load("cross_cov.fits")
        block = cross_cov[("mflike", "lensing")]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_info: dict[str, dict] = {}
        # Per-block bandpower identities, keyed by the block's (row, col) tuple.
        # Each value is ``(row_ids, col_ids)``. Two blocks may label the same
        # component in different orders; each canonicalises independently.
        self._block_ids_map: dict[tuple, tuple] = {}

    @staticmethod
    def _axis_gather(
        block_ids, target, name: str, which: str, n_block: int
    ) -> np.ndarray:
        """Index array that maps a block axis into `target`.

        `target` is either a sequence of identity keys (identity mode) or an int
        (positional mode). `n_block` is the length of the block axis being
        mapped. The decision is per axis:
          - target ids + block ids   -> gather by identity (realign + trim)
          - no target ids + no block  -> positional (size must match)
          - target ids + no block     -> raise (refuse to guess)
          - no target ids + block ids -> raise (can't map labels onto positions)
        """
        positional_target = isinstance(target, int)

        if block_ids is None:
            if positional_target:
                if target != n_block:
                    raise ValueError(
                        f"the {which} axis of a covariance block for '{name}' has "
                        f"length {n_block} but the target size is {target}; an "
                        f"unlabelled block must already match the target size."
                    )
                return np.arange(n_block)
            raise ValueError(
                f"the {which} axis of a covariance block for '{name}' does not "
                f"carry bandpower identities, but the data does; pass ids so it "
                f"can be aligned — positional alignment is unsafe because the "
                f"data is reordered to canonical order."
            )

        if positional_target:
            raise ValueError(
                f"the {which} axis of a covariance block for '{name}' carries "
                f"bandpower identities but the target order does not; cannot map "
                f"identities onto unknown positions."
            )

        if len(block_ids) != n_block:
            raise ValueError(
                f"the {which} axis of a covariance block for '{name}' has length "
                f"{n_block} but carries {len(block_ids)} ids; ids must describe the "
                f"block as stored."
            )

        position = {key: i for i, key in enumerate(block_ids)}
        if len(position) != len(block_ids):
            raise ValueError(
                f"covariance block ids for '{name}' are not unique; cannot align."
            )
        try:
            return np.array([position[key] for key in target], dtype=int)
        except KeyError as exc:
            raise ValueError(
                f"covariance for '{name}' is missing bandpower {exc.args[0]!r} "
                f"present in the data; built on a different bandpower set."
            ) from None

    def to_canonical(self, order: dict) -> np.ndarray:
        """Assemble the full joint covariance in the given per-component order.

        `order` maps component name -> target ids (canonical; already scale-cut
        for a trimmed matrix) OR an int size (positional, when the data has no
        ids). Missing blocks are left as zeros. Realign and trim fuse into one
        gather per axis.
        """
        names = list(order.keys())
        sizes = {
            n: (order[n] if isinstance(order[n], int) else len(order[n])) for n in names
        }
        starts, s = {}, 0
        for n in names:
            starts[n] = s
            s += sizes[n]
        total = s
        full = np.zeros((total, total))

        for ni in names:
            for nj in names:
                block = self.get((ni, nj))
                if block is None:
                    rev = self.get((nj, ni))
                    if rev is None:
                        continue
                    block = np.asarray(rev).T
                else:
                    block = np.asarray(block)
                row_ids, col_ids = self._block_ids((ni, nj))
                rows = self._axis_gather(row_ids, order[ni], ni, "row", block.shape[0])
                cols = self._axis_gather(col_ids, order[nj], nj, "col", block.shape[1])
                sub = block[np.ix_(rows, cols)]
                full[
                    starts[ni] : starts[ni] + sizes[ni],
                    starts[nj] : starts[nj] + sizes[nj],
                ] = sub
        return full

    @staticmethod
    def _check_ids(ids, dim, name):
        if ids is None:
            return None
        ids = list(ids)
        if len(ids) != dim:
            raise ValueError(
                f"ids for '{name}' have length {len(ids)}, expected {dim} "
                f"(one per row/col of the block as stored)."
            )
        return ids

    def _add_block(self, row, col, block, row_ids, col_ids):
        """Shared core: store a labelled covariance block (and its transpose for
        off-diagonal blocks). The single place blocks + ids are written."""
        block = np.asarray(block)
        row_ids = self._check_ids(row_ids, block.shape[0], row)
        col_ids = self._check_ids(col_ids, block.shape[1], col)
        self[(row, col)] = block
        self._block_ids_map[(row, col)] = (row_ids, col_ids)
        if row != col:
            self[(col, row)] = block.T
            self._block_ids_map[(col, row)] = (col_ids, row_ids)

    def add_component(self, name, cov, ids=None):
        """Register a component's auto-covariance (diagonal block)."""
        if isinstance(cov, dict):
            raise TypeError(f"cov must be a numpy array, not a dict. Got: {type(cov)}")
        cov_array = np.asarray(cov)
        self._add_block(name, name, cov_array, ids, ids)
        self._component_info[name] = {"size": cov_array.shape[0], "cov": cov_array}

    def add_cross_covariance(self, name1, name2, cross_cov, ids1=None, ids2=None):
        """Register the cross term between two components (off-diagonal block)."""
        self._add_block(name1, name2, cross_cov, ids1, ids2)

    def component_ids(self, name):
        """Derived per-component ids: the diagonal block's, else any block's."""
        diag = self._block_ids_map.get((name, name))
        if diag is not None and diag[0] is not None:
            return diag[0]
        for (a, b), (ia, ib) in self._block_ids_map.items():
            if a == name and ia is not None:
                return ia
            if b == name and ib is not None:
                return ib
        return None

    def _block_ids(self, key: tuple) -> tuple:
        return self._block_ids_map.get(key, (None, None))

    @property
    def component_names(self) -> list[str]:
        """Get ordered list of component names."""
        return list(self._component_info.keys())

    def _infer_component_info(self):
        """Ensure every component appearing in a block is registered.

        Explicit :meth:`add_component` entries are authoritative and kept as-is
        (and in order); any component that appears *only* in cross blocks is
        added with its size inferred from the block shape and ``cov=None`` (a
        non-auto component whose auto-covariance is supplied at assembly time).
        This is additive, so it is safe to call whether components were added
        explicitly, only via cross-covariances, or a mix of both. Raises if the
        stored blocks imply inconsistent sizes for a component.
        """
        sizes: dict[str, int] = {}

        for (name1, name2), cov in self.items():
            for name, n in ((name1, cov.shape[0]), (name2, cov.shape[1])):
                if name in sizes and sizes[name] != n:
                    raise ValueError(
                        f"Inconsistent sizes for component '{name}': "
                        f"{sizes[name]} vs {n}"
                    )
                sizes[name] = n

        for name, size in sizes.items():
            if name not in self._component_info:
                self._component_info[name] = {
                    "size": size,
                    "cov": self.get((name, name)),
                }

    def _canonical_component_ids(self, names):
        """One canonical id order per component, used to lay out the saved file.

        Blocks may label the same component in different orders; as long as they
        share the same *set* of identity keys the store can reconcile them by
        identity (this is what :meth:`to_canonical` does), so a reference order
        is chosen -- the component's own auto/diagonal block when it carries
        ids, otherwise the first labelled block seen. Raises only when two blocks
        carry genuinely *different sets* of keys, which no reordering can fix.
        Returns ``{name: ids_or_None}``.
        """
        out = {}
        for name in names:
            # Prefer the diagonal block's order as the canonical reference.
            diag = self._block_ids_map.get((name, name))
            chosen = list(diag[0]) if diag is not None and diag[0] is not None else None
            for (a, b), (ia, ib) in self._block_ids_map.items():
                for who, ids in ((a, ia), (b, ib)):
                    if who != name or ids is None:
                        continue
                    if chosen is None:
                        chosen = list(ids)
                    elif list(ids) != chosen and set(ids) != set(chosen):
                        raise ValueError(
                            f"component '{name}' has blocks labelling different "
                            f"bandpower sets; they cannot be reconciled into a "
                            f"single covariance (this is not a mere reordering)."
                        )
            out[name] = chosen
        return out

    def _warn_reordered_blocks(self, comp_ids):
        """Warn for each stored block whose labelling of a component differs from
        the canonical order, so the realignment :meth:`save` performs is never
        silent. Each physical block is reported once (its transpose is skipped).
        """
        seen = set()
        for (row, col), (row_ids, col_ids) in self._block_ids_map.items():
            pair = frozenset((row, col))
            if pair in seen:
                continue
            seen.add(pair)
            for who, ids in ((row, row_ids), (col, col_ids)):
                canon = comp_ids.get(who)
                if ids is None or canon is None or list(ids) == list(canon):
                    continue
                # Same set is guaranteed here (a set mismatch already raised).
                pos = {key: i for i, key in enumerate(ids)}
                perm = [pos[key] for key in canon]
                warnings.warn(
                    f"CrossCov.save: block {(row, col)} stored component "
                    f"'{who}' in a different order than the canonical one; "
                    f"realigning it by identity (canonical rows taken from "
                    f"stored positions {perm}).",
                    UserWarning,
                    stacklevel=3,
                )

    def save(self, path: str):
        if not path.endswith((".fits", ".sacc")):
            raise ValueError("Only .fits or .sacc files are supported!")

        # Additive, so mixed stores (some autos explicit, some components only
        # in cross blocks) are saved in full rather than truncated.
        self._infer_component_info()

        comp_ids = self._canonical_component_ids(self.component_names)
        self._warn_reordered_blocks(comp_ids)

        cross_sacc = sacc.Sacc()
        cross_sacc.metadata["component_names"] = json.dumps(self.component_names)
        cross_sacc.metadata["component_ids"] = json.dumps(comp_ids)
        cross_sacc.metadata["auto_components"] = json.dumps(
            [n for n in self.component_names if (n, n) in self]
        )

        for name in self.component_names:
            cross_sacc.add_tracer("misc", name, quantity="generic", spin=0)
        for name in self.component_names:
            for i in range(self._component_info[name]["size"]):
                cross_sacc.add_data_point("generic", (name, name), 0.0, ell=float(i))

        # Realign every block to the canonical order by identity; unlabelled
        # components fall back to positional (size) ordering.
        save_order = {
            name: (
                comp_ids[name]
                if comp_ids[name] is not None
                else self._component_info[name]["size"]
            )
            for name in self.component_names
        }
        full_cov = self.to_canonical(save_order)
        cross_sacc.add_covariance(full_cov)
        cross_sacc.save_fits(path, overwrite=True)

    def _build_full_covariance(self) -> np.ndarray:
        """Build the full joint covariance matrix from stored blocks."""
        names = self.component_names
        sizes = [self._component_info[name]["size"] for name in names]
        total_size = sum(sizes)

        full_cov = np.zeros((total_size, total_size))

        # Fill in blocks
        row_start = 0
        for i, name_i in enumerate(names):
            col_start = 0
            for j, name_j in enumerate(names):
                key = (name_i, name_j)
                if key in self:
                    block = np.asarray(self[key])
                    full_cov[
                        row_start : row_start + sizes[i],
                        col_start : col_start + sizes[j],
                    ] = block
                col_start += sizes[j]
            row_start += sizes[i]

        return full_cov

    @classmethod
    def load(cls, path: str | None) -> Optional["CrossCov"]:
        if path is None:
            return None
        if not path.endswith((".fits", ".sacc")):
            raise ValueError("Only .fits or .sacc files are supported!")

        cross_sacc = sacc.Sacc.load_fits(path)
        if "component_ids" not in cross_sacc.metadata:
            raise ValueError(
                "this cross-covariance file carries no bandpower identities "
                "(old format); regenerate it with the current CrossCov.save."
            )

        if "component_names" in cross_sacc.metadata:
            component_names = json.loads(cross_sacc.metadata["component_names"])
        else:
            component_names = list(cross_sacc.tracers.keys())

        raw = json.loads(cross_sacc.metadata["component_ids"])
        ids_map = {
            name: ([_to_hashable(c) for c in ids] if ids is not None else None)
            for name, ids in raw.items()
        }
        auto_components = set(
            json.loads(
                cross_sacc.metadata.get("auto_components", json.dumps(component_names))
            )
        )

        cross_cov = cls()
        component_indices = {
            name: cross_sacc.indices(tracers=(name, name)) for name in component_names
        }
        for name in component_names:
            cross_cov._component_info[name] = {
                "size": len(component_indices[name]),
                "cov": None,
            }

        if cross_sacc.covariance is not None:
            full_cov = cross_sacc.covariance.covmat
            for ni in component_names:
                for nj in component_names:
                    if ni == nj and ni not in auto_components:
                        continue  # no real auto block; assembly falls back to d.cov
                    block = full_cov[np.ix_(component_indices[ni], component_indices[nj])]
                    cross_cov[(ni, nj)] = block
                    cross_cov._block_ids_map[(ni, nj)] = (
                        ids_map.get(ni),
                        ids_map.get(nj),
                    )
                    if ni == nj:
                        cross_cov._component_info[ni]["cov"] = block

        return cross_cov


class MultiGaussianData(GaussianData):
    """Combined Gaussian data from multiple components with cross-covariances.

    Assembles multiple ``GaussianData`` objects into a single joint data vector
    with a combined covariance matrix that includes both auto-covariances and
    cross-covariances between components.

    A thin caller over the :class:`CrossCov` store: the ``data_list`` is the
    single source of truth for order. Each component's target order is its data
    vector's own ids (already scale-cut), and the cross-covariance blocks are
    aligned to it by identity via :meth:`CrossCov.to_canonical`. Alignment is
    decided per axis: an identity gather when both the target and the block carry
    ids; positional alignment only when neither does; and a raise when the data
    carries ids but a block does not (it refuses to guess, since the data is
    reordered to canonical order). Missing auto blocks fall back to each
    likelihood's own ``cov``, which is already in canonical data order.

    See ``.claude/plans/2026-06-05-crosscov-labelled-blocks-design.md`` for the
    design rationale.

    Parameters
    ----------
    data_list : list of GaussianData
        Individual data objects to combine
    cross_covs : CrossCov, optional
        Labelled-block cross-covariance store. If None, components are assumed
        independent. Auto-covariances can come from either the CrossCov or the
        individual GaussianData objects (individual data is used whenever the
        CrossCov doesn't contain an auto-covariance for a component).

    Attributes
    ----------
    data_list : list of GaussianData
        The original individual data objects
    names : list of str
        Names of all components
    lengths : list of int
        Data vector lengths for each component
    labels : list of str
        Component name for each element in the combined data vector

    Examples
    --------
    Combining two datasets with cross-covariance::

        data1 = GaussianData("mflike", x1, y1, cov1)
        data2 = GaussianData("lensing", x2, y2, cov2)

        cross_cov = CrossCov()
        cross_cov.add_cross_covariance("mflike", "lensing", cross_block)

        multi_data = MultiGaussianData([data1, data2], cross_cov)

        # Access combined properties
        print(multi_data.cov.shape)  # (n1 + n2, n1 + n2)
        loglike = multi_data.loglike(theory_vector)
    """

    def __init__(self, data_list, cross_covs=None):
        if cross_covs is None:
            cross_covs = CrossCov()
        self.cross_covs = cross_covs
        self.data_list = data_list
        self.lengths = [len(d) for d in data_list]
        self.names = [d.name for d in data_list]
        self._data = None

    @staticmethod
    def _kept_order(d):
        """Target identity order for a component: its data vector's own ids.

        When ``d.ids`` already describes the data vector element-for-element
        (``len(d.ids) == len(d)``) it IS the target order — e.g. mflike, whose
        data vector is already cut, so no further trimming applies. When
        ``d.ids`` spans a wider full range alongside a kept mask of the same
        length (SOLikeT scale cuts: ``len(d.ids) == len(d.indices)``), the
        target is the kept subset. Falls back to the positional size when there
        are no usable ids.
        """
        if d.ids is None:
            return len(d)
        ids = list(d.ids)
        if len(ids) == len(d):
            return ids
        if d.indices is not None and len(d.indices) == len(ids):
            kept = [k for k, keep in zip(ids, d.indices) if keep]
            if len(kept) == len(d):
                return kept
        return len(d)

    @property
    def data(self) -> GaussianData:
        if self._data is None:
            self._assemble_data()
        return self._data

    def loglike(self, theory: np.ndarray) -> float:
        return self.data.loglike(theory)

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def inv_cov(self) -> np.ndarray:
        return self.data.inv_cov

    @property
    def cov(self) -> np.ndarray:
        return self.data.cov

    @property
    def norm_const(self) -> float:
        return self.data.norm_const

    @property
    def labels(self) -> list[str]:
        return [
            x
            for y in [[name] * len(d) for name, d in zip(self.names, self.data_list)]
            for x in y
        ]

    def _assemble_data(self):
        order = {d.name: self._kept_order(d) for d in self.data_list}

        # Ensure every diagonal exists: fall back to each likelihood's own cov,
        # which is already in canonical (kept) data order.
        work = CrossCov()
        work.update(self.cross_covs)
        work._block_ids_map = dict(getattr(self.cross_covs, "_block_ids_map", {}))
        for d in self.data_list:
            if (d.name, d.name) not in work:
                work[(d.name, d.name)] = d.cov
                tgt = self._kept_order(d)
                ids = tgt if isinstance(tgt, list) else None
                work._block_ids_map[(d.name, d.name)] = (ids, ids)

        x = np.concatenate([d.x for d in self.data_list])
        y = np.concatenate([d.y for d in self.data_list])
        cov = work.to_canonical(order)
        self._data = GaussianData(" + ".join(self.names), x, y, cov)

    def plot_cov(self, **kwargs):
        import matplotlib.pyplot as plt

        labels = [
            f"{label}: {value:.2f}" for label, value in zip(self.labels, self.data.x)
        ]

        x_indices = np.arange(len(labels) + 1)
        y_indices = np.arange(len(labels) + 1)

        _, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.pcolormesh(
            x_indices, y_indices, self.cov, cmap="viridis", shading="auto"
        )

        ax.set_xticks(x_indices[:-1] + 0.5)
        ax.set_yticks(y_indices[:-1] + 0.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        ax.invert_yaxis()

        plt.colorbar(heatmap, ax=ax)

        plt.show()

        return heatmap
