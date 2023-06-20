# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
from __future__ import annotations
import numpy as np
from scipy.interpolate import splrep, BSpline  # noqa
from scipy.signal import savgol_filter  # noqa
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as clr  # type: ignore
from plotting_utils import plot_corrheatmap, plot_snr_analysis

colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_IBM = clr.LinearSegmentedColormap.from_list(
    "Blue-light cb-IBM", colors_IBM[:-1], N=256
)


class HD_Data:
    """
    Class for high dimensional data X, over a continous domain d,
    with associated response y, for regression in high dimensions.
    The class is required for the nullspace analysis.

    You need to create two instances of this class, one for the training data
    and one for the test data.
    """

    def __init__(self, X: np.ndarray, d: np.ndarray, y: np.ndarray, **kwargs):
        self.X = X
        self.d = d
        self.y = y
        self = self.standardscaler(scale_y=True, scale_x=False)
        self = self.standardscaler(scale_y=False, scale_x=True)
        self.arguments = kwargs

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
        """Add white Gaussian noise to the data.

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
        # Snippet below partly copied/adapted/inspired by:
        # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        # Answer from Noel Evans, accessed: 18.05.2022, 15:37 CET
        rows, columns = self.X.shape
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
            axs[0], self.d, self.X, cmap, x_label, r"$|\rho|$ Columns"
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
        ax.plot(self.d, np.abs(np.mean(self.X, axis=0)), label="|Mean|", lw=2.5, color=c1)
        ax.plot(
            self.d,
            np.abs(np.median(self.X, axis=0)),
            "--",
            label="|Median|",
            lw=2.5,
            color=c3,
        )
        ax.plot(self.d, np.std(self.X, axis=0), "-.", label="Std.", lw=2.5, color=c2)
        ax.legend(loc=2)
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        return ax

    def analyze_snr_by_splines(
        self,
        *,
        s: float = 0.000001,
        k: int = 5,
        mode: str = "X_raw",
        plot_snr: bool = True,
        plot_i: int = None,
        **kwargs,
    ) -> None:
        """
        Assuming we can estimate the noise by fitting a BSpline to the data, we can
        estime the SNR. This assumes that the underlying function is smooth and that
        can be reasonaby approximated by a spline.

        The spline parameter s should be chosen such that the spline fits the data according the
        desired smoothness. The default value is 0.000001 which works well for the LFP data, but
        need to be adjusted for other data.

        Arguments
        ---------
        s : float, default=0.000001
            Smoothing factor for the spline fit.
        k : int, default=5
            Degree of the spline fit.
        mode : str, default="X_raw"
            Data to be analyzed. Can be "X_raw" or "X_std".
            X_std is experimental. The mean is added back to the data before calculating the SNR.
            It is debatable whether this makes sence, but it's an interesting experiment.
        plot_snr : bool, default=True
            Plot the SNR analysis.
        plot_i : int, default=None
            Plot the spline fit for the i-th battery.
        **kwargs : dict
            additional arguments passed to calc_snr function.
        """

        if mode == "X_raw":
            X_ = self.X
        elif mode == "X_std":
            X_ = self.X_std + self.X.mean(axis=0)
        else:
            raise ValueError("Mode must be 'X_raw' or 'X_std'.")

        X_spline = np.empty(self.X.shape)

        # Loop over all the batteries and fit a spline to the data
        for i in range(X_.shape[0]):
            tck = splrep(x=self.d, y=X_[i, :], s=s, k=k)
            X_spline[i, :] = BSpline(*tck)(self.d)

        snr, noise_power = calc_snr(X_spline, X_)
        plot_snr_analysis(X_, snr, noise_power, x=self.d, s=s, title="", **kwargs)

        if plot_i is not None:
            X_i = X_[plot_i, :]
            X_spline_i = X_spline[plot_i, :]
            fig, ax = plt.subplots()
            ax.plot(self.d, X_i, label="X")
            ax.plot(self.d, X_spline_i, label="X_spline")
            if "x_label" in kwargs:
                ax.set_xlabel(kwargs["x_label"])
            else:
                ax.set_xlabel("Continous Domain, d")
            ax.set_ylabel("X")
            ax.set_title(f"X_i and X_i_spline, i={plot_i}")
            ax.legend()
            plt.show()

        if mode == "X_raw":
            self.X_spline = X_spline
            self.snr = snr
            self.snr_dB = 10 * np.log10(snr)
            self.noise_power = noise_power
        elif mode == "X_std":
            self.X_spline_std = X_spline
            self.snr_std = snr
            self.snr_dB_std = 10 * np.log10(snr)
            self.power_noise_std = noise_power

    def smooth_snr(self, *, window_length: int = 51, polyorder: int = 5) -> None:
        snr = self.snr
        self.snr_smooth = savgol_filter(snr, window_length, polyorder)
        self.snr_smooth_dB = savgol_filter(10 * np.log10(snr), window_length, polyorder)


def calc_snr(
    X_spline: np.ndarray,
    X: np.ndarray,
    *,
    method: str = "A",
) -> np.ndarray:
    """Ideas for methods from: https://github.com/hrtlacek/SNR/blob/main/SNR.ipynb"""
    Power_signal = np.mean(X_spline, axis=0) ** 2
    Power_signal_with_noise = np.mean(X, axis=0) ** 2
    Power_noise = np.mean((X - X_spline) ** 2, axis=0)

    if method == "A":
        snr = (Power_signal_with_noise - Power_noise) / Power_noise
    elif method == "B":
        snr = Power_signal / Power_noise
    elif method == "C":
        snr = np.mean(X_spline, axis=0) / np.std(X_spline, axis=0)
    else:
        raise ValueError("Method must be 'A', 'B' or 'C', read docstring for more info.")
    return snr, Power_noise
