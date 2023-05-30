from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as clr  # type: ignore
from basis_function import BasisFunction, PolynomBasis, RadialBasis
from plotting_utils import plot_corrheatmap

colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_IBM = clr.LinearSegmentedColormap.from_list(
    "Blue-light cb-IBM", colors_IBM[:-1], N=256
)


class HD_Data:
    """Class that contains all methods to manipulate the data."""

    def __init__(self, x: np.ndarray, **kwargs):
        self.x = x
        self.arguments = kwargs

    def set_X_y(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self = self.standardscaler(scale_y=True, scale_x=False)
        self = self.standardscaler(scale_y=False, scale_x=True)

    def set_X_y_from_BasicsData(self, obj: HD_Data) -> None:
        self.X = obj.X
        self.y = obj.y
        self = self.standardscaler(scale_y=True, scale_x=False)
        self = self.standardscaler(scale_y=False, scale_x=True)

    def standardscaler(self, *, scale_y: bool = True, scale_x: bool = True) -> HD_Data:
        if scale_x:
            self.stdx = np.std(self.X, axis=0)
            self.meanx = np.mean(self.X, axis=0)
            self.X_ = self.X - self.meanx
            self.X_std = self.X_ / self.stdx

        if scale_y:
            self.stdy = np.std(self.y)
            self.meany = np.mean(self.y)
            self.y_ = self.y - self.meany
            self.y_std = self.y_ / self.stdy

        return self

    def add_wgn(self, snr_x: float = None, snr_y: float = None) -> HD_Data:
        """Generates synthethic data to test the linearization methodology.

        Arguments
        ---------
        snr_x : int, default=None
            Signal to noise ratio of AWGN to be added on the signal
        snr_y : int, default=None
            Signal to noise ratio of AWGN to be added on the signal,

        Returns
        -------
        self
        """
        # Add Gaussian noise to the measurements
        # Snippet below partly copied/adapted/inspired by:
        # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        # Answer from Noel Evans, accessed: 18.05.2022, 15:37 CET
        # Calculate signal power and convert to dB

        rows, columns = self.X.shape
        # X
        if snr_x is not None:
            for i in range(rows):
                row_i = self.X[i, :]
                sig_avg_watts = np.mean(row_i**2)
                sig_avg_db = 10 * np.log10(sig_avg_watts)
                # Calculate noise according to [2] then convert to watts
                noise_avg_db = sig_avg_db - snr_x
                noise_avg_watts = 10 ** (noise_avg_db / 10)
                # Generate an sample of white noise
                mean_noise = 0
                noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), columns)
                # Noise up the original signal
                self.X[i, :] += noise

            # Update the mean centered & std data
            self = self.standardscaler(scale_y=False, scale_x=True)

        if snr_y is not None:
            for i, yi in enumerate(self.y):
                sig_avg_watts = np.mean(yi**2)
                sig_avg_db = 10 * np.log10(sig_avg_watts)
                # Calculate noise according to [2] then convert to watts
                noise_avg_db = sig_avg_db - snr_y
                noise_avg_watts = 10 ** (noise_avg_db / 10)
                # Generate an sample of white noise
                mean_noise = 0
                noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), 1)
                # Noise up the original signal
                self.y[i] += noise

            # Update the mean centered & std data
            self = self.standardscaler(scale_y=True, scale_x=False)

        return self

    def plot_row_column_corrheatmap(
        self,
        x_label: str,
        y_label: str,
        axs: plt.axes = None,
        *,
        cmap: bool = None,
    ) -> plt.axes:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        if cmap is None:
            cmap = cmap_IBM
        axs[0] = plot_corrheatmap(
            axs[0], self.x, self.X, cmap, x_label, r"$|\rho|$ Columns"
        )
        axs[1] = plot_corrheatmap(
            axs[1],
            np.arange(self.X.T.shape[1]),
            self.X.T,
            cmap,
            y_label,
            r"$|\rho|$ Rows",
            cols=False,
        )
        return axs

    def plot_stats(
        self, ax: plt.axes, c1: str, c2: str, c3: str, labelx: str, labely: str
    ) -> plt.axes:
        ax.plot(self.x, np.abs(np.mean(self.X, axis=0)), label="|Mean|", lw=2.5, color=c1)
        ax.plot(
            self.x,
            np.abs(np.median(self.X, axis=0)),
            "--",
            label="|Median|",
            lw=2.5,
            color=c3,
        )
        ax.plot(self.x, np.std(self.X, axis=0), "-.", label="Std.", lw=2.5, color=c2)
        ax.legend(loc=2)
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        return ax


def construct_data(
    x_min: float,
    x_max: float,
    basis_function: BasisFunction,
    mean_params: np.ndarray,
    stdv_params: np.ndarray,
    num_datapoints: int = 50,
    draws: int = 10,
    plot_results: bool = False,
) -> HD_Data:
    """Build an object of the basis class based on the passed parameters and return the basis object.

    Parameters
    ----------
    x_min : float
        Minimum of data range.
    x_max : float
        Maximum of data range.
    basis_function : callable (basis.function)
        Basis function defined in basis.py that it used for generating the data.
    mean_params : ndarray of shape (n_params for basis function)
        Array of means of random paramters that are used by the basis functions.
        The random parameters are drawn from normal distribution.
    stdv_params : ndarray of shape (n_params for basis function)
        Array of standarddeviations of random paramters that are used by the basis functions.
    num_datapoints : int, default=50
        Number of linearly spaced datapoints that will range from x_min to x_max
    draws : int, default=10
        Number of draws from basis functions.
    plot_results : bool, default=False
        If True, plot the data matrix.
        IF False, do not plot.
    """
    # num_basis = len(mean_params)
    x = np.linspace(x_min, x_max, num_datapoints)[:, None]
    obj = HD_Data(basis_function, x)

    # Draw the parameters for the matrix
    # m = np.random.uniform(low=range_m[0], high=range_m[1], size=rows)
    param_vals = np.zeros((draws, len(mean_params)))
    for i, (j, k) in enumerate(zip(mean_params, stdv_params)):
        param_vals[:, i] = np.array(
            [np.random.normal(loc=j, scale=k) for p in range(draws)]
        )

    # Construct it
    obj = obj.construct_X_data(param_vals)
    if plot_results:
        # Plot it# Construct it
        plt.plot(x, obj.X.T)
        plt.title("Data Generated from Basis Function")
        plt.show()

    return obj
