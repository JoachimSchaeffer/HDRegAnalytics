from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as clr  # type: ignore
from basis_function import BasisFunction, PolynomBasis, RadialBasis
from plotting_utils import plot_corrheatmap
from typing import Callable

colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_IBM = clr.LinearSegmentedColormap.from_list(
    "Blue-light cb-IBM", colors_IBM[:-1], N=256
)


class BasicsData:
    """Class that contains all methods to manipulate the data."""

    def __init__(
        self,
        function: BasisFunction,
        x: np.ndarray,
        **kwargs,
    ):
        """Initilize object
        Parameters
        ----------
        function : func
        X : ndarray
            2D numpy array of data
        x : ndarray
            1D numpy array, representing the domain values small x corresponding to each column in X
        y : ndarray
            1D array of responses
        """

        self.arguments = kwargs
        self.function = function
        self.number = function.num_basis
        self.x = x
        self.Phi_vals = self.function(x)

    def set_X_y(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self = self.standardscaler(scale_y=True, scale_x=False)
        self = self.standardscaler(scale_y=False, scale_x=True)

    def standardscaler(self, *, scale_y: bool = True, scale_x: bool = True) -> BasicsData:
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

    def construct_X_data(self, basis_weights: np.ndarray) -> BasicsData:
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

        self = self.standardscaler(scale_y=False, scale_x=True)
        return self

    def construct_y_data(
        self, targetfun: Callable[[np.ndarray], float], per_range: list = None
    ) -> BasicsData:
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

        Returns
        -------
        y : ndarray
            1D array of target/response values
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
        self = self.standardscaler(scale_y=True, scale_x=False)
        return self

    def add_wgn(self, snr_x: float = None, snr_y: float = None) -> BasicsData:
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

    # A bunch of plotting functions
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


# Helper functions for this notebook
def construct_data(
    x_min: float,
    x_max: float,
    basis_function: BasisFunction,
    mean_params: np.ndarray,
    stdv_params: np.ndarray,
    num_datapoints: int = 50,
    draws: int = 10,
    plot_results: bool = False,
) -> BasicsData:
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

    Returns
    -------
    basis_obj : object
        Initialized object containig the data X.

    Raises
    ------
    ValueError
        If objtype string does not match one of the implemented options.
    """
    # num_basis = len(mean_params)
    x = np.linspace(x_min, x_max, num_datapoints)[:, None]
    obj = BasicsData(basis_function, x)

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


def construct_plot_data_interactive(
    x_min: float,
    x_max: float,
    basis_function: BasisFunction,
    mean_param0: float,
    mean_param1: float,
    mean_param2: float,
    stdv_params0: float,
    stdv_params1: float,
    stdv_params2: float,
    num_datapoints: int = 50,
    draws: int = 10,
) -> None:
    """Wraper around 'construct_plot_data' to interact with ipython widget"""
    mean_params = np.array([mean_param0, mean_param1, mean_param2])
    stdv_params = np.array([stdv_params0, stdv_params1, stdv_params2])

    _ = construct_data(
        x_min,
        x_max,
        basis_function,
        mean_params,
        stdv_params,
        num_datapoints=num_datapoints,
        draws=draws,
        plot_results=True,
    )
    return None
