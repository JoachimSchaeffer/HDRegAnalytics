# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
from matplotlib import cm  # type: ignore
import matplotlib.cm as cmx  # type: ignore

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

from utils import optimize_pls  # type: ignore
from utils import optimize_rr  # type: ignore
from utils import nrmse  # type: ignore

from typing import Union


def plot_x_tt2(
    X: np.ndarray,
    x: np.ndarray,
    ax: plt.axes,
    color: str,
    labelx: str,
    labely: str,
    label_data: str = "Training",
    zorder: int = 1,
    **kwargs,
) -> plt.axes:
    if "linestyle" in kwargs:
        linestyle = kwargs["linestyle"]
    else:
        linestyle = "-"
    ax.plot(x, X[:, :].T, linestyle, label=label_data, lw=1, color=color, zorder=zorder)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=3)
    # axs.set_title('Training Data')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    return ax


def plot_corrheatmap(
    ax: plt.axes,
    x: np.ndarray,
    X: np.ndarray,
    cmap: cm,
    label: str,
    title: str,
    cols: bool = True,
) -> plt.axes:
    if cols:
        X_df = pd.DataFrame(X[:, ::10])
        x = x[::10]
    else:
        X_df = pd.DataFrame(X[:, :])
    X_corr = np.abs(X_df.corr())
    if cols:
        X_corr.set_index(np.round(x, 1), inplace=True)
        X_corr.set_axis(np.round(x, 1), axis="columns", inplace=True)
    mask = np.triu(X_corr)
    if cols:
        ax = sns.heatmap(
            X_corr,
            vmin=0,
            vmax=1,
            center=0.4,
            cmap=cmap,
            square=True,
            xticklabels=100,
            yticklabels=100,
            ax=ax,
            mask=mask,
        )
    else:
        ax = sns.heatmap(
            X_corr,
            vmin=0.82,
            vmax=1,
            center=0.91,
            cmap=cmap,
            square=True,
            xticklabels=10,
            yticklabels=10,
            ax=ax,
            mask=mask,
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_yticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(label)
    return ax


def optimize_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_comps: int = 20,
    alpha_lim: list = None,
    folds: int = 10,
    nb_stds: int = 1,
    plot_components: bool = False,
    std: bool = False,
    stdv: np.ndarray = None,
    min_distance_search: bool = False,
    featlin: float = 0,
    algorithm: str = "PLS",
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Crossvalidation or optimization of regression coefficient distance for PLS or RR"""
    if alpha_lim is None:
        alpha_lim = [10e-5, 10e3]

    if std:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if stdv is None:
            stdv = scaler.scale_

    if algorithm == "PLS":
        res_dict = optimize_pls(
            X,
            y,
            max_comps=max_comps,
            folds=folds,
            nb_stds=nb_stds,
            plot_components=plot_components,
            min_distance_search=min_distance_search,
            featlin=featlin,
            verbose=verbose,
            **kwargs,
        )

    elif algorithm == "RR":
        res_dict = optimize_rr(
            X,
            y,
            alpha_lim=alpha_lim,
            folds=folds,
            nb_stds=nb_stds,
            min_distance_search=min_distance_search,
            featlin=featlin,
            verbose=verbose,
            **kwargs,
        )

    if kwargs.get("plot", False):
        key = "components" if algorithm == "PLS" else "alphas"
        plot_cv_results(res_dict, key=key)
    return res_dict


def plot_cv_results(res_dict: dict, key: str = "components") -> None:
    """Plot the results of the cross validation function"""
    colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(
        res_dict["cv_res"][key],
        res_dict["cv_res"]["rmse_vals"],
        color="blue",
        label="RMSE",
    )
    ax[0].scatter(
        res_dict["cv_res"][key],
        res_dict["cv_res"]["rmse_vals"] + res_dict["cv_res"]["rmse_std"],
        color="k",
        label="RMSE + STD",
    )
    ax[0].scatter(
        res_dict["cv_res"][key],
        res_dict["cv_res"]["rmse_vals"] - res_dict["cv_res"]["rmse_std"],
        color="k",
    )

    ax[0].scatter(
        res_dict["cv_res"]["rmse_min_param"],
        np.min(res_dict["cv_res"]["rmse_vals"]),
        color=colors_IBM[1],
        s=100,
        label="RMSE Min",
    )

    # Scatter a circle around the mean rmse that is still within 1 std of the minimum
    ax[0].scatter(
        res_dict["cv_res"]["rmse_std_min_param"],
        res_dict["cv_res"]["rmse_std_min"],
        color=colors_IBM[2],
        s=100,
        label="RMSE within 1 STD of Min",
    )

    ax[0].set_xlabel(f"Number of {key}")
    ax[0].set_ylabel("RMSE")
    ax[0].set_title(f"RMSE vs. Number of {key}")
    ax[0].legend()

    ax[1].scatter(
        res_dict["l2_distance_res"][key],
        res_dict["l2_distance_res"]["l2_distance"],
        color=colors_IBM[0],
        label="L2 Distance",
    )

    min_l2_alpha = res_dict["l2_distance_res"]["l2_min_param"]
    min_l2_dist = np.min(res_dict["l2_distance_res"]["l2_distance"])
    ax[1].scatter(
        min_l2_alpha,
        min_l2_dist,
        color=colors_IBM[1],
        s=100,
        label=f"Min. L2 Dist. {min_l2_dist:.2f} {key} {min_l2_alpha:.2f}",
    )
    ax[1].set_xlabel(f"Number of {key}")
    ax[1].set_ylabel("L2 Distance")
    ax[1].set_title(f"L2 Distance vs. Number of {key}")
    ax[1].legend()

    if key == "alphas":
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
        ax[0].set_xlabel("RR alpha")
        ax[1].set_xlabel("RR alpha")
    plt.tight_layout()
    plt.show()


def truncate_colormap(
    cmap: matplotlib.cm, minval: float = 0.0, maxval: float = 1.0, n: int = 100
) -> matplotlib.cm:
    """Truncates a colormap. This is important because a) many people are partly colorblind and a lot of
    colormaps unsuited for them, and b) a lot of colormaps include yellow which can be hard to see on some
    screens and bad quality prints.
    from https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

    Parameters
    ----------
    camp : object
        matplotlib colormap object
    minval : float, default 0.0
        lower cutoff of colorbar as a fraction in range 0, 1
    maxval : float, default 1.0
        higher cutoff of colorbar as a fraction in range 0, 1
    n : int, default 100
        number linearly spaced colors that shall be placed in the colorbar

    Returns
    -------
    new_cmap : object
        new matplotlib colorbar
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def format_label(max_gamma: float, con_val: float = -9999, method: str = "") -> str:
    """Helps with label formatting for the nullspace!

    Parameters
    ----------
    max_gamma : float
        Maximum gamma value, determined by the optimization
    con_val : float, default=-9999
        Value of the constraint, default -9999 to make it easy to spot issues
        corresponding to the respective method
    method : str, default=''
        Method used to determine gamma based in the constraint value

    Returns
    -------
    label : str
        Formatted label
    """

    # Gamma highly depends on the magnitude of the difference of the features/regression coefficients
    # Thus it is very difficult to interpret.
    if method == "NRMSE":
        if con_val <= 10 ** (-12):
            label = r"$\in\mathcal{\mathbf{N}}(\mathbf{X})$"
        else:
            label = (
                r"$\in\mathcal{\mathcal{\widetilde{N}}}(\mathbf{X})$ , $\gamma\approx$"
                + f"{dec_sci_switch(max_gamma, decimal_switch=1, sci_acc=1)}, "
                + r"$\Delta_{NRMSE}\approx$"
                + f"{con_val:.2f}%"
            )
    elif method == "Xv":
        label = (
            r"$\in\mathcal{\mathcal{\widetilde{N}}}(\mathbf{X})$ , $\gamma\approx$"
            + f"{dec_sci_switch(max_gamma, decimal_switch=1)}, "
            + r"$\tilde{n}\approx$"
            + f"{con_val:.1f}%"
        )

    return label


def dec_sci_switch(number: float, decimal_switch: int = 3, sci_acc: int = 2) -> str:
    """Switch between decimal and scientific notation"""
    if number < 10 ** (-decimal_switch):
        return f"{number:.{sci_acc}e}"
    elif number > 1000:
        return f"{number:.2e}"
    else:
        return f"{number:.{decimal_switch}f}"


def plot_nullspace_analysis(
    w_alpha: np.ndarray,
    w_beta: np.ndarray,
    v: np.ndarray,
    gs: np.ndarray,
    X: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    name: str = "",
    coef_name_alpha: str = "",
    coef_name_beta: str = "",
    return_fig: bool = True,
    max_gamma: float = -9999,
    con_val: float = -9999,
    method: str = "",
) -> Union[tuple[plt.figure, plt.axes], None]:
    """Plot the nullspace correction"""
    y = y - np.mean(y)
    X = X - np.mean(X, axis=0)
    color_list = ["#0051a2", "#97964a", "#f4777f", "#93003a"]
    nrmse_vals = [
        100
        * mean_squared_error(y, X @ (w_alpha + v[-1, :]), squared=False)
        / (np.max(y) - np.min(y)),
        100
        * mean_squared_error(y, X @ (w_alpha + v[0, :]), squared=False)
        / (np.max(y) - np.min(y)),
    ]
    nrmse_min = np.min(nrmse_vals)
    nrmse_max = np.max(nrmse_vals)
    cNorm = mcolors.Normalize(vmin=nrmse_min, vmax=nrmse_max)

    cmap = truncate_colormap(cm.get_cmap("plasma"), 0.1, 0.7)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    figsize = [11, 13]
    fig, ax = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    ax[0].plot(x, X[:, :].T, label="Train", lw=1, color=color_list[0])
    ax[0].set_title("Data")
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), loc=2)
    ax[0].set_ylabel("y")

    y_min = ax[0].get_ylim()[0]
    y_max = ax[0].get_ylim()[1]
    ax[0].vlines(0, y_min, y_max, colors="k", linestyles="solid", linewidths=0.8)
    ax[0].hlines(0, min(x), max(x), colors="k", linestyles="solid", linewidths=0.8)
    ax[0].set_ylim(y_min, y_max)

    # Initializing the nrmse error to 2, makes it easy to spot issues
    nrmse = 2 * np.ones(v.shape[0])
    for i in range(v.shape[0]):
        nrmse[i] = mean_squared_error(y, X @ (w_alpha + v[i, :]), squared=False) / (
            np.max(y) - np.min(y)
        )
        ax[1].plot(
            x, w_alpha + v[i, :], color=scalarMap.to_rgba(100 * nrmse[i]), zorder=i
        )

    markevery = int(len(x) / 15)
    nrmse_alpha = (
        100
        * mean_squared_error(y, X @ (w_alpha), squared=False)
        / (np.max(y) - np.min(y))
    )
    nrmse_beta = (
        100 * mean_squared_error(y, X @ (w_beta), squared=False) / (np.max(y) - np.min(y))
    )

    coef_alpha_label = f"{coef_name_alpha}, NRMSE: {nrmse_alpha:.3f}%"
    coef_beta_label = f"{coef_name_beta}, NRMSE: {nrmse_beta:.3f}%"
    ax[1].plot(
        x,
        w_alpha,
        label=coef_alpha_label,
        color="darkgreen",
        marker="P",
        markevery=markevery,
        markersize=8,
        linewidth=2.5,
        zorder=v.shape[0] + 1,
    )
    ax[1].plot(
        x, w_beta, label=coef_beta_label, color="k", linewidth=2.5, zorder=v.shape[0] + 1
    )

    label = format_label(max_gamma, con_val=con_val, method=method)

    ax[1].fill_between(
        x.reshape(-1),
        w_alpha,
        y2=w_alpha + v[-1, :],
        color="darkgrey",
        zorder=-1,
        alpha=0.8,
        label=label,
    )

    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"Regression Coefficients $(\beta)$")

    # Set bottom and left spines as x and y axes of coordinate system
    y_min = ax[1].get_ylim()[0]
    y_max = ax[1].get_ylim()[1]
    ax[1].vlines(0, y_min, y_max, colors="k", linestyles="solid", linewidths=0.8)
    ax[1].hlines(0, min(x), max(x), colors="k", linestyles="solid", linewidths=0.8)

    ax[0].set_xlim(min(x), max(x))
    ax[1].set_ylim(y_min, y_max)
    ax[1].set_title("Nullspace Perspective")

    cb = fig.colorbar(cm.ScalarMappable(norm=cNorm, cmap=cmap), ax=ax[1], pad=0.01)
    cb.set_label(r"NRMSE (%)", labelpad=10)

    ax[0].grid()
    ax[1].grid()
    ax[1].legend(loc=2)
    fig.suptitle(name)
    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None


def plot_X(
    X: np.ndarray,
    x: np.ndarray,
    ax0_title: str = "Training Data",
    ax1_title: str = "Z-Scored",
) -> tuple[plt.figure, plt.axes]:
    stdx = np.std(X, axis=0)
    meanx = np.mean(X, axis=0)
    X_ = X - meanx
    X_std = X_ / stdx

    color_list = ["#0051a2", "#97964a", "#f4777f", "#93003a"]
    figsize = [11, 13]
    fig, ax = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
    ax[0].plot(x, X_.T, label="Train", lw=1, color=color_list[0])
    ax[0].set_title(ax0_title)
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), loc=2)
    ax[0].set_ylabel(r"$\Delta \widetilde{Q}_{100-10}$ (Ah)")

    ax[1].plot(x, X_std.T, label="Train", lw=1, color=color_list[0])
    ax[1].set_title(ax1_title)
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys(), loc=2)
    ax[1].set_ylabel(r"$\Delta \widetilde{Q}_{100-10}^{std}$")
    ax[1].set_xlim(min(x), max(x))
    ax[1].set_xlabel("Voltage (V)")
    return fig, ax


def scatter_predictions(
    X: np.ndarray,
    y_: np.ndarray,
    w: list,
    labels: list,
    title: str = "",
    ax: plt.axes = None,
    return_fig: bool = False,
) -> Union[tuple[plt.figure, plt.axes], None]:
    """Method that scatter plots the predictions associated with different regression coefficients."""

    colors = ["#000000", "#648fff", "#dc267f", "#785ef0", "#fe6100", "#ffb000"]
    # ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']
    markers = ["<", "v", "^", "o"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        fig = ax.get_figure()

    for i, (w_, label) in enumerate(zip(w, labels)):
        y_pred = X @ w_
        nrmse_ = nrmse(y_, y_pred)
        ax.scatter(
            y_,
            y_pred,
            label=f"{label}, NRMSE: {nrmse_:.2f}%",
            marker=markers[i],
            color=colors[i],
        )

    ax.set_xlabel(r"$y-\bar{y}$")
    ax.set_ylabel(r"$\hat y-\bar{y}$")
    ax.set_title(title)

    legend_fontsize = matplotlib.rcParams["legend.fontsize"]
    ax.legend(frameon=False, fontsize=legend_fontsize - 4)

    if return_fig:
        return fig, ax
    else:
        return None
