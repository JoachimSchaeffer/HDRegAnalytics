# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as clr  # type: ignore
from plotting_utils import plot_corrheatmap

colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
cmap_IBM = clr.LinearSegmentedColormap.from_list(
    "Blue-light cb-IBM", colors_IBM[:-1], N=256
)


class HD_Data:
    """
    Class for high dimensional data X, over a continous domain x,
    with associated response y, for regression in high dimensions.
    The class is required for the nullspace analysis.
    """

    def __init__(self, X: np.ndarray, x: np.ndarray, y: np.ndarray, **kwargs):
        self.X = X
        self.x = x
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
