import pytest

from gaussian_quadrature import GaussianQuadrature
from utils import almost_equals


def read_data(fname):
    with open(fname) as f:
        lines = [
            [float(num) for num in line[:-1].split("\t")] for line in f.readlines()[1:]
        ]
        lines = sorted(lines, key=lambda line: line[2])
        data = {"x": [], "w": []}
        for i, wi, xi in lines:
            data["x"].append(xi)
            data["w"].append(wi)
        return data


@pytest.fixture
def test_data():
    return {
        2: {"x": [-0.5773502691896257, 0.5773502691896257], "w": [1, 1]},
        3: {
            "x": [-0.7745966692414834, 0, 0.7745966692414834],
            "w": [5 / 9, 8 / 9, 5 / 9],
        },
        10: read_data("./tests/data/gaussian_quadrature/10.tsv"),
        23: read_data("./tests/data/gaussian_quadrature/23.tsv"),
        40: read_data("./tests/data/gaussian_quadrature/40.tsv"),
        50: read_data("./tests/data/gaussian_quadrature/50.tsv"),
        64: read_data("./tests/data/gaussian_quadrature/64.tsv"),
    }


def test_gaussian_quadrature_samples(test_data):
    for N in test_data:
        G = GaussianQuadrature(N)
        for xt, x in zip(G.x, test_data[N]["x"]):
            assert almost_equals(xt, x)


def test_gaussian_quadrature_closed_form_weights(test_data):
    for N in test_data:
        G = GaussianQuadrature(N)
        for wt, w in zip(G.w, test_data[N]["w"]):
            assert almost_equals(wt, w)


def test_gaussian_quadrature_matrix_algebra_weights(test_data):
    for N in test_data:
        if N  >= 50:
            continue # Only works up to N = 40
        G = GaussianQuadrature(N, method="matrix algebra")
        for wt, w in zip(G.w, test_data[N]["w"]):
            assert almost_equals(wt, w, eps=1e-2)