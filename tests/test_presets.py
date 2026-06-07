"""Tests for soliket.presets — the notebook-onboarding layer.

The loader turns the single-source Fiducial map (a grouped spec) into a flat
Cobaya ``params`` dict. A *dual parameter* (one carrying a ``prior``) is fixed to
its central value unless it appears in the explicit ``sample`` list.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from cobaya.theories.camb.camb import CAMB
from cobaya.tools import resolve_packages_path
from mflike import TTTEEE, BandpowerForeground

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


def test_build_info_classy_swaps_boltzmann_keeping_other_theories():
    info = build_info("mflike", theory="classy")

    assert "classy" in info["theory"]
    assert "camb" not in info["theory"]
    assert "mflike.BandpowerForeground" in info["theory"]  # non-Boltzmann preserved


def test_build_info_honors_params_dir_override(tmp_path):
    (tmp_path / "cosmo.yaml").write_text("ns: {value: 0.5, latex: 'n_s'}\n")

    info = build_info("lensing", params_dir=str(tmp_path))

    assert info["params"]["ns"]["value"] == 0.5


def test_build_info_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        build_info("nope")


def test_session_exposes_roles_fiducial_and_loglike():
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
def test_quickstart_builds_runnable_session(preset, expect_mflike, expect_lensing):
    s = quickstart(preset)

    assert s.cosmo is not None
    assert (s.mflike is not None) is expect_mflike
    assert (s.lensing is not None) is expect_lensing
    assert np.isfinite(s.loglike())


def test_resolve_aliases_finds_roles_by_class_regardless_of_order():
    cosmo, fg, mflike = _bare(CAMB), _bare(BandpowerForeground), _bare(TTTEEE)
    # deliberately scrambled order
    model = SimpleNamespace(components=[fg, mflike, cosmo])

    roles = resolve_aliases(model)

    assert roles.mflike is mflike
    assert roles.foreground is fg
    assert roles.cosmo is cosmo
    assert roles.lensing is None


def test_resolve_aliases_recurses_into_multigaussian():
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

    params = load_params(params_dir=str(tmp_path))

    assert params["ns"]["value"] == 0.5            # overridden file used
    assert params["a_tSZ"]["value"] == 3.30        # foreground fell back to bundled
    assert params["cal_LAT_93"]["value"] == 1      # systematics fell back to bundled


def test_load_params_rejects_non_mapping_override(tmp_path):
    # A malformed hand-edited override (here: a YAML list) must fail with a clear
    # error naming the file, not a cryptic AttributeError deep in build_params.
    bad = tmp_path / "cosmo.yaml"
    bad.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="cosmo.yaml"):
        load_params(params_dir=str(tmp_path))


def test_build_params_errors_on_missing_group():
    # A preset requesting a group with no params/<group>.yaml must fail clearly,
    # naming the missing group, not raise a bare KeyError.
    with pytest.raises(ValueError, match="systematics"):
        build_params({"cosmo": {}}, groups=["cosmo", "systematics"])


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
