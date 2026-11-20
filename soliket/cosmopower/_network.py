"""
Numpy-only reader for trained CosmoPower networks.

SOLikeT only evaluates CosmoPower emulators, it never trains them, and evaluation is a
plain numpy forward pass. Reading the ``.npz`` files written by ``cosmopower`` here
keeps ``cosmopower`` (and TensorFlow) out of SOLikeT's dependencies. The forward pass
reproduces ``cosmopower_NN.forward_pass_np`` and
``cosmopower_PCAplusNN.forward_pass_np``.
"""

import os
from dataclasses import dataclass

import numpy as np

KINDS = ("NN", "PCAplusNN")


def _read_only(array) -> np.ndarray:
    array = np.array(array)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class PCABasis:
    """PCA basis that maps the network output onto the (normalised) features."""

    mean: np.ndarray  # (n_pcas,)
    std: np.ndarray  # (n_pcas,)
    matrix: np.ndarray  # (n_pcas, n_modes)


@dataclass(frozen=True)
class CosmoPowerNetwork:
    """
    A trained CosmoPower network (``NN`` or ``PCAplusNN``), for inference only.

    The network standardises its inputs, applies ``len(weights) - 1`` dense layers
    with the CosmoPower activation
    :math:`(\\beta + (1 - \\beta)\\,\\sigma(\\alpha z))\\,z`, a final linear layer,
    optionally a PCA projection, and rescales the result to the output features.
    """

    parameters: tuple[str, ...]
    modes: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    alphas: tuple[np.ndarray, ...]
    betas: tuple[np.ndarray, ...]
    parameters_mean: np.ndarray
    parameters_std: np.ndarray
    features_mean: np.ndarray
    features_std: np.ndarray
    pca: PCABasis | None = None

    def __post_init__(self):
        n_layers = len(self.weights)
        if len(self.biases) != n_layers:
            raise ValueError(
                f"Got {n_layers} weight matrices but {len(self.biases)} biases."
            )
        if len(self.alphas) != n_layers - 1 or len(self.betas) != n_layers - 1:
            raise ValueError(
                f"A network with {n_layers} layers needs {n_layers - 1} alphas and betas,"
                f" got {len(self.alphas)} and {len(self.betas)}."
            )
        if self.weights[0].shape[0] != len(self.parameters):
            raise ValueError(
                f"First layer takes {self.weights[0].shape[0]} inputs but the network"
                f" lists {len(self.parameters)} parameters."
            )
        for i in range(1, n_layers):
            if self.weights[i].shape[0] != self.weights[i - 1].shape[1]:
                raise ValueError(
                    f"Weights of layer {i} have shape {self.weights[i].shape}, which does"
                    f" not chain with layer {i - 1} of shape {self.weights[i - 1].shape}."
                )
        n_out = self.weights[-1].shape[1]
        if self.pca is not None:
            if self.pca.matrix.shape[0] != n_out:
                raise ValueError(
                    f"The network outputs {n_out} PCA coefficients but the PCA basis has"
                    f" {self.pca.matrix.shape[0]} components."
                )
            n_out = self.pca.matrix.shape[1]
        if len(self.modes) != n_out:
            raise ValueError(
                f"The network outputs {n_out} features but lists {len(self.modes)} modes."
            )

    @property
    def kind(self) -> str:
        """The CosmoPower network type, ``"NN"`` or ``"PCAplusNN"``."""
        return "NN" if self.pca is None else "PCAplusNN"

    @classmethod
    def from_npz(cls, filename: str, kind: str | None = None) -> "CosmoPowerNetwork":
        """
        Load a network saved by ``cosmopower`` in its ``.npz`` format.

        :param filename: path to the network, with or without the ``.npz`` suffix.
        :param kind: expected network type (``"NN"`` or ``"PCAplusNN"``). The type is
                     read from the file; if given, it must match.
        :return: the loaded network, with read-only arrays.
        """
        if kind is not None and kind not in KINDS:
            raise ValueError(f"Unknown network type {kind!r}, expected one of {KINDS}.")

        base = filename[: -len(".npz")] if filename.endswith(".npz") else filename
        if not os.path.exists(base + ".npz"):
            if os.path.exists(base + ".pkl"):
                raise FileNotFoundError(
                    f"Found only the pickled network {base}.pkl. SOLikeT reads the .npz"
                    " format: convert it once with cosmopower, by restoring the network"
                    " and calling its save method."
                )
            raise FileNotFoundError(f"CosmoPower network {base}.npz does not exist.")

        with np.load(base + ".npz") as f:
            n_layers = int(f["n_layers"])
            pca = None
            if "pca_transform_matrix" in f.files:
                pca = PCABasis(
                    mean=_read_only(f["pca_mean"]),
                    std=_read_only(f["pca_std"]),
                    matrix=_read_only(f["pca_transform_matrix"]),
                )
            network = cls(
                parameters=tuple(str(p) for p in f["parameters"]),
                modes=_read_only(f["modes"]),
                weights=tuple(_read_only(f[f"W_{i}"]) for i in range(n_layers)),
                biases=tuple(_read_only(f[f"b_{i}"]) for i in range(n_layers)),
                alphas=tuple(_read_only(f[f"alphas_{i}"]) for i in range(n_layers - 1)),
                betas=tuple(_read_only(f[f"betas_{i}"]) for i in range(n_layers - 1)),
                parameters_mean=_read_only(f["parameters_mean"]),
                parameters_std=_read_only(f["parameters_std"]),
                features_mean=_read_only(f["features_mean"]),
                features_std=_read_only(f["features_std"]),
                pca=pca,
            )

        if kind is not None and kind != network.kind:
            raise ValueError(
                f"Network {base}.npz is a {network.kind} network, but type {kind!r} was"
                " requested."
            )
        return network

    def predictions_np(self, parameters: dict) -> np.ndarray:
        """
        Evaluate the network.

        :param parameters: values (scalars or 1D arrays of equal length) for at least
                           every name in :attr:`parameters`; extra keys are ignored.
        :return: array of shape ``(n_samples, n_modes)``.
        """
        x = np.stack([np.atleast_1d(parameters[p]) for p in self.parameters], axis=1)
        x = (x - self.parameters_mean) / self.parameters_std
        hidden = zip(self.weights[:-1], self.biases[:-1], self.alphas, self.betas)
        for W, b, alpha, beta in hidden:
            z = x @ W + b
            x = (beta + (1.0 - beta) / (1.0 + np.exp(-alpha * z))) * z
        x = x @ self.weights[-1] + self.biases[-1]
        if self.pca is not None:
            x = (x * self.pca.std + self.pca.mean) @ self.pca.matrix
        return x * self.features_std + self.features_mean

    def ten_to_predictions_np(self, parameters: dict) -> np.ndarray:
        """:math:`10^{\\rm output}`, for networks trained on log-spectra."""
        return 10.0 ** self.predictions_np(parameters)
