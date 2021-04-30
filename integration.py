"""A file for all the integration methods"""
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

class IntegrationMD(Integration):
    def __init__(self, M, method, N, func, a=-1, b=1) -> None:
        grid, weightsMD = create_M_dimensional_weights(M, method, N, a, b)
        self.weights = np.reshape(weightsMD, weightsMD.size)
        if type(func) == np.ndarray:
            self.func_values = np.reshape(func, func.size)
        else:
            func_values_md = np.zeros([N + 1 for _ in range(M)])
            for idx, _ in np.ndenumerate(func_values_md):
                func_values_md[idx] = func(grid[idx])
            self.func_values = np.reshape(func_values_md, func_values_md.size)
        super().__init__(self.weights, self.func_values)

def create_M_dimensional_weights(M, method, N, a, b):
    shape = [N + 1 for _ in range(M)]
    get_a = lambda i: a[i] if hasattr(a, '__iter__') else a
    get_b = lambda i: b[i] if hasattr(b, '__iter__') else b
    get_method = lambda i: method[i] if hasattr(method, '__iter__') else method
    individual_integrals = [get_method(i)(N, None, a=get_a(i), b=get_b(i)) for i in range(M)]
    weights = 1
    for i in range(M - 1, -1, -1):
        weights = np.outer(individual_integrals[i].weights, weights).reshape(shape[i:])
    inputs = np.array([integral.xs for integral in individual_integrals])
    grid = np.ones(shape + [M])
    for idx, value in np.ndenumerate(inputs):
        m, k = idx
        grid[(slice(None),) * m + (k,) + (slice(None),) * (M - m - 1) + (m,)] = value
    return grid, weights

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
    # print("------------- Integration of x^2 --------------------")
    # N = 600
    # a = -6
    # b = 2
    # f = lambda x: x * x
    # integrals = [
    #     GaussianQuadratureIntegration(N, f, a=a, b=b, method="closed form").value,
    #     TrapezoidRule(N, f, a=a, b=b).value,
    #     SimpsonsRule(N, f, a=a, b=b).value,
    #     Simpsons38Rule(N, f, a=a, b=b).value,
    #     BoolesRule(N, f, a=a, b=b).value,
    # ]
    # print(integrals)
    # print("------------- 2D Integration of x^2 + y^2 --------------------")
    # N = 600
    # f = lambda x, y: x*x + y*y
    # integral = Integration2D(GaussianQuadratureIntegration, BoolesRule, N, f) # going over (-1, 1) in x and y
    # print(integral)
    # print("------------- 2D Integration of x^2*y^2 --------------------")
    # N = 600
    # f = lambda x, y: x*x * y*y
    # integral = Integration2D(GaussianQuadratureIntegration, BoolesRule, N, f) # going over (-1, 1) in x and y
    # print(integral)
    # print("------------- 2D Integration of y*cos(pi*xy) --------------------")
    # N = 600
    # f = lambda x, y: y*np.cos(np.pi*x*y)
    # integral = Integration2D(GaussianQuadratureIntegration, BoolesRule, N, f) # going over (-1, 1) in x and y
    # print(integral)
    # print("------------- 1D Integration of -2cos(y)/y --------------------")
    # N = 600
    # # f = lambda y: -2 * np.cos(y) / y
    # f = lambda x: np.exp(1j*x)
    # integral = GaussianQuadratureIntegration(N, f)
    # print(integral)
    # print("------------- 2D Integration of exp(ixy) --------------------")
    # N = 600
    # f = lambda x, y: np.exp(1j * x * y)
    # integral = Integration2D(GaussianQuadratureIntegration, GaussianQuadratureIntegration, N, f) # going over (-1, 1) in x and y
    # print(integral)
    print("-------------- ND Integration ------------------------------")
    f = lambda xs: np.sum(np.array(xs) ** 2)
    N = 52
    integral = IntegrationMD(4, BoolesRule, N, f)
    print(integral) # should be 21.33333333333