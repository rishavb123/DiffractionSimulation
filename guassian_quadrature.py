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
            method (str, optional): The method used to calculate the weights. Could be matrix algebra, integration, or closed form. Defaults to "closed form".

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
        self.calcualte_weights()

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
        # TODO: come back to the integration method after finishing trapezoind and simpsons rule
        # need to build out trapezoid and simpsons rule in a seperate integration file to finish this function
        return self.__matrix_algebra()

    def __closed_form(self):
        coefficients = one_hot(self.N, self.N + 1)
        w = [
            1
            / (1 - xk * xk)
            * (
                np.polynomial.legendre.legval(
                    xk, np.polynomial.legrendre.legder(coefficients)
                )
            )
            ** -2
            for xk in self.x
        ]
        return w