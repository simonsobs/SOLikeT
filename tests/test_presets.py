"""Tests for soliket.presets — the notebook-onboarding layer.

The loader turns the single-source Fiducial map (a grouped spec) into a flat
Cobaya ``params`` dict. A *dual parameter* (one carrying a ``prior``) is fixed to
its central value unless it appears in the explicit ``sample`` list.
"""

import os
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from cobaya.theories.camb.camb import CAMB
from cobaya.tools import resolve_packages_path

from soliket.gaussian import MultiGaussianLikelihood
from soliket.lensing import LensingLikelihood
from soliket.presets import (
    Session,
    build_info,
    build_params,
    load_params,
    quickstart,
    resolve_aliases,
)


def _mflike_data_available():
    path = resolve_packages_path()
    return bool(path) and os.path.isfile(
        os.path.join(path, "data", "MFLike", "v0.8", "LAT_simu_sacc_00044.fits")
    )


def _bare(cls):
    """A real-typed instance without running the heavy __init__."""
    return cls.__new__(cls)


def test_build_info_mflike_wires_likelihood_theory_and_params():
    info = build_info("mflike")

    assert "mflike.TTTEEE" in info["likelihood"]
    assert "camb" in info["theory"]
    assert "mflike.BandpowerForeground" in info["theory"]
    # params come from the Fiducial map, all fixed by default
    assert info["params"]["tau"]["value"] == 0.0544
    assert info["params"]["a_tSZ"]["value"] == 3.30


def test_build_info_lensing_includes_only_cosmo_params():
    info = build_info("lensing")

    assert "soliket.LensingLikelihood" in info["likelihood"]
    assert info["params"]["ns"]["value"] == 0.9649
    assert "a_tSZ" not in info["params"]  # foreground not wired for lensing-only


def test_build_info_threads_sample_into_params():
    info = build_info("mflike", sample=["tau"])

    assert "prior" in info["params"]["tau"]
    assert info["params"]["ns"]["value"] == 0.9649  # unlisted stays fixed


def test_build_info_defaults_to_camb_theory():
    assert "camb" in build_info("mflike")["theory"]


def test_build_info_camb_uses_single_massive_neutrino():
    info = build_info("mflike")
    ea = info["theory"]["camb"]["extra_args"]
    assert ea["num_massive_neutrinos"] == 1
    assert ea["nnu"] == 3.044
    # mnu is now an ordinary cosmo param (lives in cosmo.yaml), not a bare float
    assert info["params"]["mnu"]["value"] == 0.06
    # the camb precision extra_args survive the overlay
    assert ea["lens_potential_accuracy"] == 1
    # the old normal-hierarchy keys are gone from the bundled preset
    assert "nu_mass_eigenstates" not in ea


def test_build_info_neutrino_default_does_not_clobber_override(tmp_path):
    # An override that pins mnu (the ISO normal-hierarchy case) must win over
    # the injected single-massive default of 0.06.
    (tmp_path / "cosmo.yaml").write_text(
        "ns: {value: 0.9649, latex: 'n_s'}\n"
        "mnu: {value: 0.12, latex: '\\\\sum m_\\\\nu'}\n"
    )
    info = build_info("lensing", defaults_dir=str(tmp_path))
    assert info["params"]["mnu"]["value"] == 0.12


def test_build_info_rejects_unknown_theory():
    with pytest.raises(ValueError, match="unknown theory"):
        build_info("mflike", theory="cosmomc")


def test_build_info_classy_uses_classy_neutrino_subblock():
    info = build_info("mflike", theory="classy")

    assert "classy" in info["theory"]
    assert "camb" not in info["theory"]
    assert "mflike.BandpowerForeground" in info["theory"]  # non-Boltzmann preserved
    ea = info["theory"]["classy"]["extra_args"]
    assert ea["N_ncdm"] == 1
    assert ea["N_ur"] == 2.0328
    # classy-native neutrino param, with the camb-name alias
    assert info["params"]["m_ncdm"]["value"] == 0.06
    assert info["params"]["m_ncdm"]["renames"] == "mnu"
    assert "mnu" not in info["params"]  # mnu is camb-only; not injected under classy


def test_build_info_honors_defaults_dir_override(tmp_path):
    (tmp_path / "cosmo.yaml").write_text("ns: {value: 0.5, latex: 'n_s'}\n")

    info = build_info("lensing", defaults_dir=str(tmp_path))

    assert info["params"]["ns"]["value"] == 0.5


def test_build_info_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        build_info("nope")


def test_session_exposes_roles_fiducial_and_loglike(check_skip_mflike):
    from mflike import TTTEEE

    cosmo, mflike = CAMB.__new__(CAMB), TTTEEE.__new__(TTTEEE)
    fake_model = SimpleNamespace(
        components=[cosmo, mflike],
        loglikes=lambda point=None: (np.array([-1.5]), {}),
    )
    info = {"params": {"ns": {"value": 0.9649}}}

    s = Session(info, fake_model)

    assert s.model is fake_model
    assert s.info is info
    assert s.fiducial == {"ns": {"value": 0.9649}}
    assert s.mflike is mflike
    assert s.cosmo is cosmo
    assert s.lensing is None
    assert s.loglike() == -1.5


@pytest.mark.skipif(not _mflike_data_available(), reason="MFLike data not installed")
@pytest.mark.parametrize(
    "preset, expect_mflike, expect_lensing",
    [
        ("mflike", True, False),
        ("lensing", False, True),
        ("multigaussian", True, True),
    ],
)
def test_quickstart_builds_runnable_session(
    preset, expect_mflike, expect_lensing, check_skip_mflike
):
    s = quickstart(preset)

    assert s.cosmo is not None
    assert (s.mflike is not None) is expect_mflike
    assert (s.lensing is not None) is expect_lensing
    assert np.isfinite(s.loglike())


def test_resolve_aliases_finds_roles_by_class_regardless_of_order(check_skip_mflike):
    from mflike import TTTEEE, BandpowerForeground

    cosmo, fg, mflike = _bare(CAMB), _bare(BandpowerForeground), _bare(TTTEEE)
    # deliberately scrambled order
    model = SimpleNamespace(components=[fg, mflike, cosmo])

    roles = resolve_aliases(model)

    assert roles.mflike is mflike
    assert roles.foreground is fg
    assert roles.cosmo is cosmo
    assert roles.lensing is None


def test_resolve_aliases_recurses_into_multigaussian(check_skip_mflike):
    from mflike import TTTEEE

    cosmo = _bare(CAMB)
    multi = _bare(MultiGaussianLikelihood)
    multi.likelihoods = [_bare(TTTEEE), _bare(LensingLikelihood)]
    model = SimpleNamespace(components=[cosmo, multi])

    roles = resolve_aliases(model)

    assert roles.mflike is multi.likelihoods[0]
    assert roles.lensing is multi.likelihoods[1]
    assert roles.cosmo is cosmo


def test_load_params_all_fixed_by_default():
    params = load_params()

    # Dual params collapse to their fiducial central value...
    assert params["tau"]["value"] == 0.0544
    assert params["ns"]["value"] == 0.9649
    assert params["a_tSZ"]["value"] == 3.30
    # ...always-fixed and derived params survive untouched.
    assert params["T_d"]["value"] == 9.7
    assert params["omegam"]["derived"] is True


def test_load_params_attaches_cobaya_theory_renames():
    # Renames are sourced from Cobaya's own camb/classy tables so cosmo params
    # are theory-portable: e.g. ombh2 -> omega_b under CLASS via the omegabh2 alias.
    params = load_params()

    assert "omegabh2" in params["ombh2"]["renames"]
    assert "omegach2" in params["omch2"]["renames"]
    assert "omegal" in params["omega_de"]["renames"]


def test_load_params_can_restrict_to_groups():
    cosmo_only = load_params(groups=["cosmo"])

    assert "ns" in cosmo_only
    assert "a_tSZ" not in cosmo_only  # foreground group excluded
    assert "cal_LAT_93" not in cosmo_only  # systematics group excluded


def test_load_params_keeps_prior_for_sampled_params():
    params = load_params(sample=["tau", "a_tSZ"])

    assert "prior" in params["tau"]
    assert "value" not in params["tau"]
    assert "prior" in params["a_tSZ"]
    # Unlisted dual params stay fixed.
    assert params["ns"]["value"] == 0.9649


def test_load_fiducial_map_reads_all_groups():
    from soliket.presets import load_fiducial_map

    spec = load_fiducial_map()
    assert set(spec) == {"cosmo", "foreground", "systematics"}
    assert spec["cosmo"]["ns"]["ref"]["loc"] == 0.9649


def test_load_params_override_dir_replaces_one_group(tmp_path):
    # Drop in only cosmo.yaml; foreground/systematics must still come from the package.
    (tmp_path / "cosmo.yaml").write_text("ns: {value: 0.5, latex: 'n_s'}\n")

    params = load_params(defaults_dir=str(tmp_path))

    assert params["ns"]["value"] == 0.5            # overridden file used
    assert params["a_tSZ"]["value"] == 3.30        # foreground fell back to bundled
    assert params["cal_LAT_93"]["value"] == 1      # systematics fell back to bundled


def test_load_params_rejects_non_mapping_override(tmp_path):
    # A malformed hand-edited override (here: a YAML list) must fail with a clear
    # error naming the file, not a cryptic AttributeError deep in build_params.
    bad = tmp_path / "cosmo.yaml"
    bad.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="cosmo.yaml"):
        load_params(defaults_dir=str(tmp_path))


def test_build_params_errors_on_missing_group():
    # A preset requesting a group with no params/<group>.yaml must fail clearly,
    # naming the missing group, not raise a bare KeyError.
    with pytest.raises(ValueError, match="systematics"):
        build_params({"cosmo": {}}, groups=["cosmo", "systematics"])


def _mflike_fg_defaults():
    """Merge mflike's shipped foreground param defaults into one flat dict.

    The ``fg_*.yaml`` files are flat param dicts; ``Foreground.yaml`` carries a
    top-level ``params:`` block with the always-used foreground params.
    """
    merged = {}
    for spec in ("TT", "TE", "EE"):
        txt = resources.files("mflike").joinpath(f"fg_{spec}.yaml").read_text()
        merged.update(yaml.safe_load(txt) or {})
    fg = yaml.safe_load(resources.files("mflike").joinpath("Foreground.yaml").read_text())
    merged.update((fg or {}).get("params") or {})
    return merged


def test_presets_foreground_matches_mflike_defaults(check_skip_mflike):
    """Drift tripwire: presets foreground priors must match mflike's defaults.

    ``soliket/presets/params/foreground.yaml`` DUPLICATES the ``prior``,
    ``proposal`` and ``latex`` values from mflike's shipped foreground defaults
    (``fg_TT.yaml``, ``fg_TE.yaml``, ``fg_EE.yaml`` and the ``params:`` block of
    ``Foreground.yaml``), adding an SO-specific ``ref`` that mflike does not ship.
    We deliberately keep foreground explicit rather than auto-pulling from mflike,
    so this test exists purely as a tripwire: if an mflike version bump changes a
    default we copied, this fails loudly instead of silently shifting our priors.

    When it fails: review mflike's change and deliberately re-pin
    ``soliket/presets/params/foreground.yaml`` to match (or, if the deviation is
    an intentional SO choice, add a documented exclusion here). Do NOT just
    force it green.

    Only the INTERSECTION of param names is compared (SO-specific params absent
    from mflike, and mflike params absent from presets, are ignored), and only
    the ``prior``/``proposal``/``latex`` fields — the presets-only ``ref`` is
    ignored.
    """
    from soliket.presets import load_fiducial_map

    presets_fg = load_fiducial_map()["foreground"]
    mflike_fg = _mflike_fg_defaults()

    fields = ("prior", "proposal", "latex")
    mismatches = []
    for name in sorted(set(presets_fg) & set(mflike_fg)):
        p, m = presets_fg[name], mflike_fg[name]
        for field in fields:
            # Treat absence consistently: a key missing on either side reads as
            # None, so "present here / absent there" surfaces as a mismatch.
            pv, mv = p.get(field), m.get(field)
            if pv != mv:
                mismatches.append((name, field, pv, mv))

    assert not mismatches, (
        "Presets foreground priors have DRIFTED from mflike's shipped defaults. "
        "This tripwire guards against an mflike bump silently changing the "
        "foreground priors we duplicated in "
        "soliket/presets/params/foreground.yaml. Review mflike's change and "
        "deliberately re-pin that file (or add a documented exclusion). "
        "Mismatches (name, field, presets_value, mflike_value): "
        + "; ".join(
            f"{name}.{field}: presets={pv!r} mflike={mv!r}"
            for name, field, pv, mv in mismatches
        )
    )


def test_build_info_calls_are_independent():
    # The neutrino sector now comes from theory.yaml, loaded fresh per call.
    # Mutating one returned info must not leak into the next call's defaults.
    info = build_info("mflike", theory="classy")
    info["params"]["m_ncdm"]["value"] = 999
    info["theory"]["classy"]["extra_args"]["N_ncdm"] = 999

    fresh = build_info("mflike", theory="classy")
    assert fresh["params"]["m_ncdm"]["value"] == 0.06
    assert fresh["theory"]["classy"]["extra_args"]["N_ncdm"] == 1


def test_build_info_theory_dir_override_replaces_neutrino_extra_args(tmp_path):
    # ISO normal-hierarchy case: a defaults folder carrying its own theory.yaml
    # REPLACES the packaged single-massive neutrino extra_args wholesale (per-file
    # fallback), so the conflicting baseline keys never appear -- no merge, no
    # deletion needed.
    (tmp_path / "theory.yaml").write_text(
        "camb:\n"
        "  extra_args:\n"
        "    num_nu_massive: 2\n"
        "    nu_mass_eigenstates: 2\n"
        "    share_delta_neff: true\n"
    )

    info = build_info("lensing", defaults_dir=str(tmp_path))
    ea = info["theory"]["camb"]["extra_args"]

    # the NH override is present...
    assert ea["num_nu_massive"] == 2
    assert ea["nu_mass_eigenstates"] == 2
    # ...the packaged single-massive keys are gone (wholesale replacement)...
    assert "num_massive_neutrinos" not in ea
    assert "nnu" not in ea
    # ...and the preset-skeleton accuracy survives the overlay.
    assert ea["kmax"] == 0.9


def test_build_info_theory_falls_back_to_bundled_when_dir_lacks_it(tmp_path):
    # A defaults folder that overrides only cosmo must still get the packaged
    # neutrino theory.yaml (per-file fallback, mirroring the param groups).
    (tmp_path / "cosmo.yaml").write_text("ns: {value: 0.9649, latex: 'n_s'}\n")

    info = build_info("lensing", defaults_dir=str(tmp_path))
    ea = info["theory"]["camb"]["extra_args"]

    assert ea["num_massive_neutrinos"] == 1
    assert ea["nnu"] == 3.044


def test_build_info_template_dir_override_patches_likelihood_option(tmp_path):
    # A defaults folder carrying templates/<preset>.yaml overlays the packaged
    # skeleton (recursive_update): the named option changes...
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "lensing.yaml").write_text(
        "likelihood:\n"
        "  soliket.LensingLikelihood:\n"
        "    theory_lmax: 3000\n"
    )

    info = build_info("lensing", defaults_dir=str(tmp_path))
    like = info["likelihood"]["soliket.LensingLikelihood"]

    # the override wins...
    assert like["theory_lmax"] == 3000
    # ...and the rest of the packaged skeleton is inherited (overlay, not replace).
    assert info["theory"]["camb"]["extra_args"]["kmax"] == 0.9
    assert "evaluate" in info["sampler"]


def test_build_info_template_falls_back_to_bundled_when_dir_lacks_it(tmp_path):
    # A defaults folder overriding only params (no templates/) leaves the packaged
    # likelihood skeleton untouched.
    (tmp_path / "cosmo.yaml").write_text("ns: {value: 0.9649, latex: 'n_s'}\n")

    info = build_info("lensing", defaults_dir=str(tmp_path))
    assert info["likelihood"]["soliket.LensingLikelihood"]["theory_lmax"] == 5000


def _mgl_options(info):
    mgl = info["likelihood"]["soliket.gaussian.MultiGaussianLikelihood"]
    return mgl["components"], mgl["options"]


def test_multigaussian_composes_options_from_members_in_component_order():
    # The joint skeleton declares only `components`; build_info fills `options`
    # positionally from the member presets (mflike.yaml / lensing.yaml).
    info = build_info("multigaussian")
    components, options = _mgl_options(info)

    assert components == ["mflike.TTTEEE", "soliket.LensingLikelihood"]
    assert options[0]["input_file"] == "LAT_simu_sacc_00044.fits"  # from mflike.yaml
    assert options[1]["theory_lmax"] == 5000  # from lensing.yaml


def test_multigaussian_theory_unions_member_precision():
    # Regression: the joint camb extra_args must carry BOTH members' precision
    # (mflike's Transfer.* and lensing's kmax), not silently drop either.
    ea = build_info("multigaussian")["theory"]["camb"]["extra_args"]

    assert ea["kmax"] == 0.9  # from lensing.yaml
    assert ea["Transfer.kmax"] == 1.2  # from mflike.yaml
    assert ea["WantTransfer"] is True  # from mflike.yaml
    # the joint-level skeleton override is applied last (last wins)
    assert build_info("multigaussian")["theory"]["camb"]["stop_at_error"] is False
    # the foreground theory component rides in from the mflike member
    assert "mflike.BandpowerForeground" in build_info("multigaussian")["theory"]


def test_member_template_override_flows_to_standalone_and_joint(tmp_path):
    # A single folder override on a member template reaches BOTH the standalone
    # member preset and the joint preset that composes it -- so an imprint built
    # from `lensing` and a fit built from `multigaussian` stay consistent.
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "lensing.yaml").write_text(
        "likelihood:\n  soliket.LensingLikelihood:\n    theory_lmax: 3000\n"
    )

    solo = build_info("lensing", defaults_dir=str(tmp_path))
    _, joint_options = _mgl_options(build_info("multigaussian", defaults_dir=str(tmp_path)))

    assert solo["likelihood"]["soliket.LensingLikelihood"]["theory_lmax"] == 3000
    assert joint_options[1]["theory_lmax"] == 3000


def test_dual_param_fixed_to_central_when_not_sampled():
    spec = {
        "cosmo": {
            "tau": {
                "prior": {"dist": "norm", "loc": 0.0544, "scale": 0.0073},
                "ref": {"dist": "norm", "loc": 0.0544, "scale": 0.0073},
                "proposal": 0.0073,
                "latex": r"\tau",
            }
        }
    }

    params = build_params(spec, sample=[])

    assert params["tau"] == {"value": 0.0544, "latex": r"\tau"}
