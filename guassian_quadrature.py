"""A file for choosing the points of the Guassian Quadrature and calculating the weights"""
import numpy as np
from numpy.lib import polynomial
from numpy.polynomial.legendre import legder, legval

from utils import one_hot

class GuassianQuadrature:
    """The guassian quadrature class encapsulating all the methods"""

    def __init__(self, N, method="closed form") -> None:
        """Creates an object of the Guassian Quadrature class and calculates the samples and weights

        Args:
            N (int): The number of sample points to use
            method (str, optional): The method used to calculate the weights. Could be matrix algebra, integration, or closed form. Defaults to "closed form". Note that only the closed form solution will work for larger N.

        Raises:
            ValueError: The value of method was not matrix algebra, integration, or closed form.
        """
        self.N = N
        if method.lower() not in {"matrix algebra", "integration", "closed form"}:
            raise ValueError(
                "The method must be one of the three provided options: matrix algebra, integration, or closed form"
            )
        self.method = method
        self.choose_samples()
        self.calculate_weights()

    def choose_samples(self):
        """Chooses N samples between -1 and 1 by calculating the roots of a legendre polynomial of degree N.

        Returns:
            numpy.array: The array of smaple points
        """
        coefficients = one_hot(self.N, self.N + 1)
        roots = np.polynomial.legendre.legroots(coefficients)
        self.x = roots
        return roots

    def calculate_weights(self):
        """Applies the method chosen in the constructor to calculate the weight for each sample point

        Raises:
            ValueError: The value of method was not matrix algebra, integration, or closed form.

        Returns:
            np.array: The weights for each sample point
        """
        if self.method == "matrix algebra":
            self.w = self.__matrix_algebra()
        elif self.method == "integration":
            self.w = self.__integration()
        elif self.method == "closed form":
            self.w = self.__closed_form()
        else:
            raise ValueError(
                "The method must be one of the three provided options: matrix algebra, integration, or closed form"
            )
        return self.w

    def __matrix_algebra(self):
        a = np.vander(
            self.x, increasing=True
        ).transpose()  # Builds our Vandermonde matrix
        b = np.array(
            [(1 - (1 if k % 2 == 0 else -1)) / k for k in range(1, self.N + 1)]
        )  # Creates our vector of all the true integrals
        w = np.linalg.solve(a, b)
        return w

    def __integration(self):
        def phi(k):
            def f(x):
                temp = np.copy(self.x)
                t = temp[k]
                temp = np.delete(temp, k)
                arr = (x - temp) / (t - temp)
                return np.prod(arr)

            return f

        w = np.array([BoolesRule(1000, phi(k)).value for k in range(self.N)])
        return w

    def __closed_form(self):
        coefficients = one_hot(self.N, self.N + 1)
        w = np.array(
            [
                2
                / (1 - xk * xk)
                * (
                    np.polynomial.legendre.legval(
                        xk, np.polynomial.legendre.legder(coefficients)
                    )
                )
                ** -2
                for xk in self.x
            ]
        )
        return w

C=isinstance
class Integration:
	def __init__(self,weights,func):
		self.weights=weights
		if isinstance(func,np.ndarray):self.func_values=func
		elif isinstance(func,tuple):assert isinstance(func[0],np.ndarray);assert callable(func[1]);self.func_values=np.vectorize(func[1])(func[0])
		else:raise TypeError('Did not recognize the type of func')
		self.value=self.evaluate()
	def evaluate(self):self.value=np.dot(self.weights,self.func_values);return self.value
	def __str__(self):return str(self.value)
class BoolesRule(Integration):
	def __init__(self,N,func,a=-1,b=1):
		h=(b-a)/N;assert N%4==0;weights=np.ones(N+1)*h/45;weights[0]*=14;weights[-1]*=14
		for i in range(1,N):weights[i]*=64 if i%2==1 else 24 if i%4==2 else 28
		xs=np.arange(a,b+h,h);super().__init__(weights,(xs,func))