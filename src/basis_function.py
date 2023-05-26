# The following code is partly copied, extended on build on:
# https://github.com/lawrennd/mlai/
# based on commit: bb4b776a21ec17001bf20ee6406324217f902944
# expand it to the different basis funcitons in the source.
import numpy as np
from abc import ABC, abstractmethod


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
