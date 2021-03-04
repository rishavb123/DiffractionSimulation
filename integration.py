"""A file for all the integration methods"""
# TODO: comment this file
import numpy as np

from gaussian_quadrature import GaussianQuadrature


class Integration:
    def __init__(self, weights, func) -> None:
        self.weights = weights
        if func is None:
            return
        if isinstance(func, np.ndarray):
            self.func_values = func
        elif isinstance(func, tuple):
            if func[1] is None:
                return
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

class Integration2D(Integration):
    def __init__(self, method1, method2, N, func, a=-1, b=1, c=-1, d=1) -> None:
        self.int1 = method1(N, None, a=a, b=b)
        self.int2 = method2(N, None, a=c, b=d)
        weights2d = np.outer(self.int1.weights, self.int2.weights)
        self.weights = np.reshape(weights2d, weights2d.size)
        func_values = []    
        for x in self.int1.xs:
            for y in self.int2.xs:
                func_values.append(func(x, y))
        self.func_values = np.array(func_values)
        super().__init__(self.weights, self.func_values)

class GaussianQuadratureIntegration(Integration):
    def __init__(self, N, func, a=-1, b=1, method="closed form") -> None:
        self.G = GaussianQuadrature(N, method=method)
        mid = (a + b) / 2
        dist = (b - a) / 2
        if func != None:
            self.func = np.vectorize(lambda x: dist * func(dist * x + mid))
        else:
            self.func = None
        self.xs = self.G.x
        super().__init__(self.G.w, (self.G.x, self.func))


class TrapezoidRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        weights = np.ones(N + 1) * h
        weights[0] = h / 2
        weights[-1] = h / 2
        self.xs = np.arange(a, b + h, h)
        super().__init__(weights, (self.xs, func))


class SimpsonsRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 2 == 0
        weights = np.ones(N + 1) * h / 3
        for i in range(1, N):
            weights[i] *= 4 if i % 2 == 1 else 2
        self.xs = np.arange(a, b + h, h)
        super().__init__(weights, (self.xs, func))


class Simpsons38Rule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 3 == 0
        weights = np.ones(N + 1) * h / 8
        weights[0] *= 3
        weights[-1] *= 3
        for i in range(1, N):
            weights[i] *= 6 if i % 3 == 0 else 9
        self.xs = np.arange(a, b + h, h)
        super().__init__(weights, (self.xs, func))


class BoolesRule(Integration):
    def __init__(self, N, func, a=-1, b=1) -> None:
        h = (b - a) / N
        assert N % 4 == 0
        weights = np.ones(N + 1) * h / 45
        weights[0] *= 14
        weights[-1] *= 14
        for i in range(1, N):
            weights[i] *= 64 if i % 2 == 1 else (24 if i % 4 == 2 else 28)
        self.xs = np.arange(a, b + h, h)
        super().__init__(weights, (self.xs, func))


if __name__ == "__main__":
    print("------------- Integration of x^2 --------------------")
    N = 600
    a = -6
    b = 2
    f = lambda x: x * x
    integrals = [
        GaussianQuadratureIntegration(N, f, a=a, b=b, method="closed form").value,
        TrapezoidRule(N, f, a=a, b=b).value,
        SimpsonsRule(N, f, a=a, b=b).value,
        Simpsons38Rule(N, f, a=a, b=b).value,
        BoolesRule(N, f, a=a, b=b).value,
    ]
    print(integrals)
    print("------------- 2D Integration of x^2 + y^2 --------------------")
    N = 600
    f = lambda x, y: x*x + y*y
    integral = Integration2D(GaussianQuadratureIntegration, BoolesRule, N, f) # going over (-1, 1) in x and y
    print(integral)