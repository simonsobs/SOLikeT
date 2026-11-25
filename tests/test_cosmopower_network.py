"""
Tests for the numpy-only reader of trained CosmoPower networks.
"""

import os

import numpy as np
import pytest

from soliket.cosmopower import CosmoPowerNetwork, PCABasis

CP_PAPER = os.path.join(
    os.path.dirname(__file__), "..", "soliket", "cosmopower", "data", "CP_paper"
)

FIDUCIAL = {
    "omega_b": 0.0224,
    "omega_cdm": 0.122,
    "h": 0.67,
    "tau_reio": 0.065,
    "n_s": 0.9645,
    "ln10^{10}A_s": 3.07,
}


def _sigmoid_gate(z, alpha, beta):
    return (beta + (1.0 - beta) / (1.0 + np.exp(-alpha * z))) * z


def _tiny_network(pca=None, n_out=1):
    """One input, one hidden unit, values chosen to be easy to follow by hand."""
    return CosmoPowerNetwork(
        parameters=("x",),
        modes=np.arange(2, 2 + (n_out if pca is None else pca.matrix.shape[1])),
        weights=(np.array([[2.0]]), np.full((1, n_out), 3.0)),
        biases=(np.array([1.0]), np.full(n_out, 0.5)),
        alphas=(np.array([1.0]),),
        betas=(np.array([0.25]),),
        parameters_mean=np.array([1.0]),
        parameters_std=np.array([2.0]),
        features_mean=np.full(n_out if pca is None else pca.matrix.shape[1], 10.0),
        features_std=np.full(n_out if pca is None else pca.matrix.shape[1], 2.0),
        pca=pca,
    )


def test_nn_forward_pass_matches_hand_computation():
    net = _tiny_network()
    # x = 3 -> standardised 1 -> hidden z = 2*1 + 1 = 3 -> gated -> 3*h + 0.5
    hidden = _sigmoid_gate(3.0, 1.0, 0.25)
    expected = (3.0 * hidden + 0.5) * 2.0 + 10.0

    out = net.predictions_np({"x": [3.0]})

    assert out.shape == (1, 1)
    assert np.isclose(out[0, 0], expected)


def test_pca_forward_pass_matches_hand_computation():
    pca = PCABasis(
        mean=np.array([0.1, -0.2]),
        std=np.array([2.0, 0.5]),
        matrix=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
    )
    net = _tiny_network(pca=pca, n_out=2)
    hidden = _sigmoid_gate(3.0, 1.0, 0.25)
    coeffs = np.full(2, 3.0 * hidden + 0.5) * pca.std + pca.mean
    expected = (coeffs @ pca.matrix) * 2.0 + 10.0

    out = net.predictions_np({"x": [3.0]})

    assert out.shape == (1, 3)
    assert np.allclose(out[0], expected)


def test_ten_to_predictions_is_power_of_ten():
    net = _tiny_network()
    params = {"x": [0.3]}
    expected = 10.0 ** net.predictions_np(params)
    assert np.allclose(net.ten_to_predictions_np(params), expected)


def test_inputs_are_ordered_by_network_parameters():
    net = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TT_NN"))
    shuffled = dict(reversed(list(FIDUCIAL.items())))
    shuffled["unused"] = [123.0]
    assert np.array_equal(net.predictions_np(shuffled), net.predictions_np(FIDUCIAL))


def test_scalar_and_batched_inputs():
    net = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TT_NN"))
    single = net.predictions_np(FIDUCIAL)
    batched = net.predictions_np({k: [v, v] for k, v in FIDUCIAL.items()})
    assert single.shape == (1, 2507)
    assert batched.shape == (2, 2507)
    assert np.allclose(batched[1], single[0])


# Reference values computed with the cosmopower fork
# (git+https://github.com/HTJense/cosmopower, predictions_np) at FIDUCIAL, for
# output indices [0, 500, 1500, 2506], i.e. ell = [2, 502, 1502, 2508].
REFERENCE = {
    "cmb_TT_NN": ("NN", [-9.832409157, -14.086164, -15.59147865, -16.98744098]),
    "cmb_EE_NN": ("NN", [-14.2726512, -16.56525079, -17.31833142, -18.40907975]),
    "cmb_TE_PCAplusNN": (
        "PCAplusNN",
        [4.019322481e-13, -1.900810075e-16, -1.512985549e-19, -4.123881681e-19],
    ),
}


@pytest.mark.parametrize("name", list(REFERENCE))
def test_bundled_networks_match_cosmopower(name):
    kind, expected = REFERENCE[name]
    net = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, name), kind=kind)

    out = net.predictions_np(FIDUCIAL)[0]

    assert list(net.parameters) == list(FIDUCIAL)
    assert np.array_equal(net.modes[[0, -1]], [2, 2508])
    assert np.allclose(out[[0, 500, 1500, 2506]], expected, rtol=1e-6, atol=0)


def test_from_npz_accepts_explicit_suffix():
    a = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_EE_NN"))
    b = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_EE_NN.npz"))
    assert np.array_equal(a.predictions_np(FIDUCIAL), b.predictions_np(FIDUCIAL))


def test_kind_is_inferred_and_checked():
    nn = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TT_NN"))
    pca = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TE_PCAplusNN"))
    assert nn.kind == "NN" and nn.pca is None
    assert pca.kind == "PCAplusNN" and pca.pca is not None

    with pytest.raises(ValueError, match="PCAplusNN"):
        CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TE_PCAplusNN"), kind="NN")
    with pytest.raises(ValueError, match="Unknown network type"):
        CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TT_NN"), kind="GP")


def test_missing_file_and_pickle_only(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CosmoPowerNetwork.from_npz(str(tmp_path / "nope"))

    (tmp_path / "old.pkl").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="convert"):
        CosmoPowerNetwork.from_npz(str(tmp_path / "old"))


def test_loaded_arrays_are_read_only():
    net = CosmoPowerNetwork.from_npz(os.path.join(CP_PAPER, "cmb_TE_PCAplusNN"))
    with pytest.raises(ValueError):
        net.weights[0][0, 0] = 0.0
    with pytest.raises(ValueError):
        net.pca.matrix[0, 0] = 0.0


def test_inconsistent_shapes_are_rejected():
    good = _tiny_network()
    kwargs = {f: getattr(good, f) for f in good.__dataclass_fields__}

    with pytest.raises(ValueError, match="alphas"):
        CosmoPowerNetwork(**{**kwargs, "alphas": ()})
    with pytest.raises(ValueError, match="layer 1"):
        CosmoPowerNetwork(**{**kwargs, "weights": (np.ones((1, 1)), np.ones((2, 1)))})
    with pytest.raises(ValueError, match="parameters"):
        CosmoPowerNetwork(**{**kwargs, "parameters": ("x", "y")})
    with pytest.raises(ValueError, match="modes"):
        CosmoPowerNetwork(**{**kwargs, "modes": np.arange(5)})
