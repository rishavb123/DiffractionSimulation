"""A file for all the integration methods"""
# TODO: comment this file
import numpy as np

from guassian_quadrature import GuassianQuadrature


class Integration:
    def __init__(self, weights, func) -> None:
        self.weights = weights
        if isinstance(func, np.ndarray):
            self.func_values = func
        elif isinstance(func, tuple):
            assert isinstance(func[0], np.ndarray)
            assert callable(func[1])
            self.func_values = np.vectorize(func[1])(func[0])
        else:
            raise TypeError("Did not recognize the type of func")

        self.value = self.evaluate()

    def evaluate(self):
        self.value = np.dot(self.weights, self.func_values)
        return self.value

    def __str__(self) -> str:
        return str(self.value)


class GuassianQuadratureIntegration(Integration):
    def __init__(self, N, func, a=-1, b=1, method="closed form") -> None:
        self.G = GuassianQuadrature(N, method=method)
        mid = (a + b) / 2
        dist = (b - a) / 2
        self.func = np.vectorize(lambda x: dist * func(dist * x + mid))
        super().__init__(self.G.w, (self.G.x, self.func))


class TrapezoidRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        weights = np.ones(N + 1) * h
        weights[0] = h / 2
        weights[-1] = h / 2
        xs = np.arange(a, b + h, h)
        super().__init__(weights, (xs, func))


class SimpsonsRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 2 == 0
        weights = np.ones(N + 1) * h / 3
        for i in range(1, N):
            weights[i] *= 4 if i % 2 == 1 else 2
        xs = np.arange(a, b + h, h)
        super().__init__(weights, (xs, func))


class Simpsons38Rule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 3 == 0
        weights = np.ones(N + 1) * h / 8
        weights[0] *= 3
        weights[-1] *= 3
        for i in range(1, N):
            weights[i] *= 6 if i % 3 == 0 else 9
        xs = np.arange(a, b + h, h)
        super().__init__(weights, (xs, func))


class BoolesRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 4 == 0
        weights = np.ones(N + 1) * h / 45
        weights[0] *= 14
        weights[-1] *= 14
        for i in range(1, N):
            weights[i] *= 64 if i % 2 == 1 else (24 if i % 4 == 2 else 28)
        xs = np.arange(a, b + h, h)
        super().__init__(weights, (xs, func))


if __name__ == "__main__":
    N = 600
    a = -6
    b = 2
    f = lambda x: x * x
    integrals = [
        GuassianQuadratureIntegration(N, f, a=a, b=b, method="closed form").value,
        TrapezoidRule(N, f, a=a, b=b).value,
        SimpsonsRule(N, f, a=a, b=b).value,
        Simpsons38Rule(N, f, a=a, b=b).value,
        BoolesRule(N, f, a=a, b=b).value,
    ]
    print(integrals)