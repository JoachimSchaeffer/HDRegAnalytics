# The following code is partly copied, extended on build on:
# https://github.com/lawrennd/mlai/
# based on commit: bb4b776a21ec17001bf20ee6406324217f902944
# expand it to the different basis funcitons in the source.
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class BasisDataConstructor:
    def __init__(
        self,
        basis_function: BasisFunction,
        basis_weights: np.ndarray,
        targetfun: Callable[[np.ndarray], float],
        per_range: list = None,
        **kwargs,
    ):
        self.basis_function = basis_function
        self.kwargs = kwargs
        self.construct_X_data(basis_weights)
        self.construct_y_data(targetfun, per_range)

    def construct_X_data(self, basis_weights: np.ndarray) -> None:
        """Constructs a data matrix based on
        Parameters
        ----------
        basis_weights: ndarray
            2D array of rows equal to desired observations in the data matrix, cols equal num_basis
        """
        if self.number != basis_weights.shape[1]:
            raise ValueError(
                "Number of basis weights per observation must equal the number defined for this object!"
            )
        self.X = np.zeros((basis_weights.shape[0], self.Phi_vals.shape[0]))
        for i in range(basis_weights.shape[0]):
            # Iterate through the rows.
            self.X[i, :] = np.dot(self.Phi_vals, basis_weights[i, :].T)

    def construct_y_data(
        self, targetfun: Callable[[np.ndarray], float], per_range: list = None
    ) -> None:
        """Construct responsese

        Parameters
        ----------
        targetfun : callable
            Underlying relationship between X and y. This can be any function from R^n -> R^1
            This is also the ideal feature for predicting y and thus the information we would like to discover by applying the lionearization methodology.
        percentage_range_x_to_t : ndrray, default=[0,1]
            1D array with two elements, the first one being strictly smaller than the second value, both being strcitly between 0 and 1,
            defines the range of input data that shall be used to generate the target function
            The reson behind this is that in process data analytics often a sitation can arise where only a part of the data is relevant to predict the target y
        """
        if per_range is None:
            per_range = [0, 1]

        columns = self.X.shape[1]
        low_ind = int(per_range[0] * columns)
        high_ind = int(per_range[1] * columns)
        # self.y = targetfun(self.X[:, low_ind:high_ind])

        rows = self.X.shape[0]
        self.y = np.zeros([rows])
        for i in range(rows):
            row_i = self.X[i, :]
            self.y[i] = targetfun(row_i[low_ind:high_ind])


class BasisFunction(ABC):
    def __init__(self, num_basis: int, data_limits: list = None, **kwargs):
        self.num_basis = num_basis
        if data_limits is None:
            self.data_limits = [-1.0, 1.0]
        else:
            self.data_limits = data_limits
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class PolynomBasis(BasisFunction):
    def __init__(self, num_basis: int, data_limits: list = None, **kwargs):
        super().__init__(num_basis, data_limits, **kwargs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        centre = self.data_limits[0] / 2.0 + self.data_limits[1] / 2.0
        span = self.data_limits[1] - self.data_limits[0]
        z = x - centre
        z = 2 * z / span
        Phi = np.zeros((x.shape[0], self.num_basis))
        for i in range(self.num_basis):
            Phi[:, i : i + 1] = z**i
        return Phi


class RadialBasis(BasisFunction):
    def __init__(
        self, num_basis: int, data_limits: list = None, width: float = None, **kwargs
    ):
        super().__init__(num_basis, data_limits, **kwargs)
        self.width = width

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.num_basis > 1:
            centres = np.linspace(
                self.data_limits[0], self.data_limits[1], self.num_basis
            )
            if self.width is None:
                self.width = (centres[1] - centres[0]) / 2.0
        else:
            centres = np.asarray([self.data_limits[0] / 2.0 + self.data_limits[1] / 2.0])
            if self.width is None:
                self.width = (self.data_limits[1] - self.data_limits[0]) / 2.0

        Phi = np.zeros((x.shape[0], self.num_basis))
        for i in range(self.num_basis):
            Phi[:, i : i + 1] = np.exp(
                -0.5 * ((np.asarray(x, dtype=float) - centres[i]) / self.width) ** 2
            )
        return Phi


class FourierBasis(BasisFunction):
    def __init__(self, num_basis: int, data_limits: list = None, **kwargs):
        super().__init__(num_basis, data_limits, **kwargs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        tau = 2 * np.pi
        # span = float(data_limits[1] - data_limits[0])
        Phi = np.ones((x.shape[0], self.num_basis))
        for i in range(1, self.num_basis):
            if i % 2:
                Phi[:, i : i + 1] = np.sin((i + 1) * tau * np.asarray(x, dtype=float))
            else:
                Phi[:, i : i + 1] = np.cos((i + 1) * tau * np.asarray(x, dtype=float))
        return Phi
