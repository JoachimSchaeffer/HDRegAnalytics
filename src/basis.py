# The following code is partly copied, extended on build on:
# https://github.com/lawrennd/mlai/
# based on commit: bb4b776a21ec17001bf20ee6406324217f902944
# expand it to the different basis funcitons in the source.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from src.helper import plot_corrheatmap


colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_IBM = clr.LinearSegmentedColormap.from_list(
    "Blue-light cb-IBM", colors_IBM[:-1], N=256
)


class BasicsData:
    """Class that contains all methods to manipulate the data.
    Basis Data: Generate artificail data based on basis functions.
    Basic Data: Set data by insertig X and y.
    """

    def __init__(self, function=None, number=None, X=None, x=None, y=None, **kwargs):
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
        self.number = number
        self.function = function
        self.Phi_vals = None
        self.x = x

        self.X = X
        if self.X is None:
            self.X_ = None
            self.X_std = None
            self.stdx = None
            self.meanx = None
        else:
            self = self.standardscaler(scale_y=False, scale_x=True)

        self.y = y
        if self.y is None:
            self.y_ = None
            self.y_std = None
            self.stdy = None
            self.meany = None
        else:
            self = self.standardscaler(scale_y=True, scale_x=False)

    def Phi(self, x):
        """Create basis vector phi
        Parameters
        ----------
        x : ndarray
            1D array of x values where the function should be evaluated
        """
        self.Phi_vals = self.function(x, num_basis=self.number, **self.arguments)
        self.x = x
        return self

    def standardscaler(self, scale_y=True, scale_x=True):
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

    def construct_X_data(self, basis_weights):
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
            # Itereate trhough the rows.
            self.X[i, :] = np.dot(self.Phi_vals, basis_weights[i, :].T)

        self = self.standardscaler(scale_y=False, scale_x=True)
        return self

    def construct_y_data(self, targetfun, per_range: list = None):
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

    def add_wgn(self, add_noise_X=True, snr_x=50, add_noise_y=False, snr_y=50):
        """Generates synthethic data to test the linearization methodology.

        Arguments
        ---------
        add_noise_X : bool
            whether to add wgn to X
        snr_x : int, default=50
            Signal to noise ratio of AWGN to be added on the signal
        add_noise_y : bool
            whether to add wgn to y
        snr_y : int, default=50
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
        if add_noise_X:
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

        if add_noise_y:
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
    def plot_row_column_corrheatmap(self, x_label, y_label, cmap=None, axs=None):
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

    def plot_stats(self, ax, c1, c2, c3, labelx, labely):
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


# Differen basis for functions.e
def polynomial(x, num_basis=4, data_limits: list = None):
    """Polynomial basis"""

    if data_limits is None:
        data_limits = [-1.0, 1.0]

    centre = data_limits[0] / 2.0 + data_limits[1] / 2.0
    span = data_limits[1] - data_limits[0]
    z = np.asarray(x, dtype=float) - centre
    z = 2 * z / span
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i : i + 1] = z**i
    return Phi


def radial(x, num_basis=4, data_limits: list = None, width=None):
    """Radial basis constructed using exponentiated quadratic form."""

    if data_limits is None:
        data_limits = [-1.0, 1.0]

    if num_basis > 1:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis)
        if width is None:
            width = (centres[1] - centres[0]) / 2.0
    else:
        centres = np.asarray([data_limits[0] / 2.0 + data_limits[1] / 2.0])
        if width is None:
            width = (data_limits[1] - data_limits[0]) / 2.0

    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i : i + 1] = np.exp(
            -0.5 * ((np.asarray(x, dtype=float) - centres[i]) / width) ** 2
        )
    return Phi


def fourier(x, num_basis=4, data_limits: list = None):
    """Fourier basis"""

    if data_limits is None:
        data_limits = [-1.0, 1.0]

    tau = 2 * np.pi
    # span = float(data_limits[1] - data_limits[0])
    Phi = np.ones((x.shape[0], num_basis))
    for i in range(1, num_basis):
        if i % 2:
            Phi[:, i : i + 1] = np.sin((i + 1) * tau * np.asarray(x, dtype=float))
        else:
            Phi[:, i : i + 1] = np.cos((i + 1) * tau * np.asarray(x, dtype=float))
    return Phi


def relu(x, num_basis=4, data_limits: list = None, gain=None):
    """Rectified linear units basis"""

    if data_limits is None:
        data_limits = [-1.0, 1.0]

    if num_basis > 2:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis)[:-1]
    elif num_basis == 2:
        centres = np.asarray([data_limits[0] / 2.0 + data_limits[1] / 2.0])
    else:
        centres = []
    if num_basis < 3:
        basis_gap = data_limits[1] - data_limits[0]
    else:
        basis_gap = (data_limits[1] - data_limits[0]) / (num_basis - 2)
    if gain is None:
        gain = np.ones(num_basis - 1) / basis_gap
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i : i + 1] = (
            gain[i - 1]
            * (np.asarray(x, dtype=float) > centres[i - 1])
            * (np.asarray(x, dtype=float) - centres[i - 1])
        )
    return Phi


def hyperbolic_tangent(x, num_basis=4, data_limits: list = None, gain=None):
    """Hyperbolic tangents"""

    if data_limits is None:
        data_limits = [-1.0, 1.0]

    if num_basis > 2:
        centres = np.linspace(data_limits[0], data_limits[1], num_basis - 1)
        width = (centres[1] - centres[0]) / 2.0
    elif num_basis == 2:
        centres = np.asarray([data_limits[0] / 2.0 + data_limits[1] / 2.0])
        width = (data_limits[1] - data_limits[0]) / 2.0
    else:
        centres = []
        width = None
    if gain is None and width is not None:
        gain = np.ones(num_basis - 1) / width
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i : i + 1] = np.tanh(
            gain[i - 1] * (np.asarray(x, dtype=float) - centres[i - 1])
        )
    return Phi


# Helper functions for this notebook
def construct_data(
    x_min,
    x_max,
    basis_function,
    mean_params,
    stdv_params,
    num_datapoints=50,
    draws=10,
    plot_results=False,
):
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
    num_basis = len(mean_params)
    x = np.linspace(x_min, x_max, num_datapoints)[:, None]
    obj = BasicsData(basis_function, num_basis).Phi(x)

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
    x_min,
    x_max,
    basis_function,
    mean_param0,
    mean_param1,
    mean_param2,
    stdv_params0,
    stdv_params1,
    stdv_params2,
    num_datapoints=50,
    draws=10,
):
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
        plot_results=1,
    )
    return None
