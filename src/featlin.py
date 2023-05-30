# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import copy

from hd_data import HD_Data
from src.nullspace import Nullspace
from src.nullspace import format_label, nrmse

from plotting_utils import optimize_cv

import jax.numpy as jnp
from jax import jacfwd


class Featlin:
    """Class that performs feature linearization, comparison and potentially selection.
    By choice we decided not to inherit other classes in here, but instead realy on composition.
    """

    def __init__(self, X=None, x=None, y=None, data_obj=None, feat_funcs=None):
        """Initiate Featlin object.
        Either pass X, x, y, where 'x' is either the domain of your measurements (e.g. 2.0V-3.5V or Hz fro spectra)
        You can also pass a linearly spaced vecotr x, if the functional domain is less well defined.
        Only the plots will be affected by x.

        Parameters
        ----------
        feat_funcs : list of jax numpy functions
            feature functions that shall be testes and among which the algorithm will decide.

        """

        if X is None or y is None:
            if data_obj is None:
                raise ValueError(
                    "You must pass either X and y, or data_obj to instatiate class object"
                )
            else:
                self.data = data_obj
        else:
            if x is None:
                x = np.linspace(0, X.shape[1] - 1, X.shape[1])
            self.data = HD_Data(X=X, x=x, y=y)

        if feat_funcs is None:
            feat_fun = [
                lambda a: jnp.mean(a),
                lambda a: jnp.sum(a**2),
                lambda a: jnp.var(a),
                lambda x: jax_moment(x, 3) / ((jax_moment(x, 2)) ** (3 / 2)),
                lambda x: jax_moment(x, 4) / (jax_moment(x, 2) ** 2) - 3,
            ]
            feat_fun_names = [
                "Sum",
                "Sum of Squares",
                "Variance",
                "Skewness",
                "Kurtosis",
            ]
            self.feat_fun_dict = {
                feat_fun_names[i]: feat_fun[i] for i in range(len(feat_fun))
            }
        else:
            self.feat_fun_dict = feat_funcs

        self.nullspace_dict = dict.fromkeys(
            self.feat_fun_dict.keys()
        )  # Filling with results of the runs, nullspace vectors etc.
        columns = ["Featue", "Model", "CV", "min_dist", "NRMSE", "Pearson"]
        self.results = pd.DataFrame(
            columns=columns
        )  # Filling with fresults from the run. Overview!

        # Making plotting stuff a lot easier by setting some color combinations
        # colors = ['#332bb3', '#4a31b5', '#5d37b6', '#6d3db7', '#7c43b7', '#8a49b6', '#964fb5', '#a256b3', '#ad5db1', '#b764b0', '#c16cae', '#ca75ad', '#d27eac', '#d989ab', '#e094aa', '#e7a1ab', '#ecafac', '#f0beae', '#f4cfb0', '#f6e1b4']
        colors = [
            "#332288",
            "#117733",
            "#44AA99",
            "#88CCEE",
            "#DDCC77",
            "#CC6677",
            "#AA4499",
            "#882255",
        ]
        colors_IBM = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000"]
        self.cmap_ = clr.LinearSegmentedColormap.from_list(
            "Blue-light cb-safe", colors, N=256
        )
        self.cmap = clr.LinearSegmentedColormap.from_list(
            "Blue-light cb-IBM", colors_IBM[:-1], N=256
        )
        # color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
        # self.color_list = [colors_IBM[0], colors_IBM[2], colors_IBM[3], colors_IBM[4], colors_IBM[5]]
        self.color_list = colors[::-1] + colors
        self.marker_list = ["s", "o", "D", "P"]

    def regress_linearized_coeff(self, fun, std=False):
        """Estimation of m and b via OLS regression.
        std : bool
            The way this is implemented is that the facotr will only be applied after the estimation
            of the linearized coefficients. This is done because we want to comopare the coefficeints correponding to the
            standardized data. But this means, X must be provided prior to standardization!
        """

        X = self.data.X
        y = self.data.y

        x_hat = np.zeros(len(X))
        a = np.mean(X, axis=0)
        gradient = jacfwd(fun)

        for i in range(len(X)):
            # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            x_hat[i] = fun(a) + np.dot((X[i, :] - a), gradient(a))

        reg = LinearRegression(fit_intercept=True).fit(x_hat.reshape(-1, 1), y)

        m = reg.coef_
        b = reg.intercept_
        linearized_coef = m * gradient(a)
        # The constant coefficient should be equal to the mean of y
        linearized_const_coef = m * fun(a) + b
        # Test whether the constant coefficient is equal to the mean of y with a tolerance of 1% of the mean of y
        assert np.isclose(
            linearized_const_coef, np.mean(y), rtol=0.01
        ), f"Linearized constant coefficient is not equal to the mean of y \
              with a tolerance of 1% of the mean of y. Linearized constant coefficient: \
              {linearized_const_coef}, mean of y: {np.mean(y)}"

        lin_coef = np.array(linearized_coef)

        if std:
            lin_coef = lin_coef * self.data.stdx

        return x_hat, lin_coef, np.array(linearized_const_coef)

    def analyze_all_features(
        self,
        opt_cv: dict = None,
        opt_dist: dict = None,
        fig_props: dict = None,
        con_thres=0.01,
        opt_gamma_method="Xv",
        std=False,
        verbose=False,
        gray_line_params=None,
    ):
        """analyzes all Features"""
        # Create a list of labels from a.1) to f.4)
        # labels = [
        #     f"{chr(97+i)}.{j})"
        #     for i in range(len(self.nullspace_dict.keys()))
        #     for j in range(1, 5)
        # ]

        if opt_cv is None:
            opt_cv = {"active": True, "max_comp": 10}
        if opt_dist is None:
            opt_dist = {"active": False}
        if fig_props is None:
            fig_props = {"save": False, "multiple_fig": True}

        if not fig_props["multiple_fig"]:
            fig = plt.figure(
                constrained_layout=True,
                figsize=(26, 4.72 * len(self.nullspace_dict.keys())),
            )
            subfigs = fig.subfigures(nrows=len(self.nullspace_dict.keys()), ncols=1)

        for i, key in enumerate(self.nullspace_dict.keys()):
            self.analyze_feature(
                key,
                opt_cv=opt_cv,
                opt_dist=opt_dist,
                plot_cv=0,
                con_thres=con_thres,
                opt_gamma_method=opt_gamma_method,
                std=std,
                verbose=verbose,
            )
            if fig_props["multiple_fig"]:
                fig, axs = self.linearization_plot(key, std=std)
                # Set title of the figure
                # fig.suptitle(f'Linearized {key} Feature',  y=0.94)
                axs[0].set_title(f"Linearized {key} Feature")
                if fig_props["save"]:
                    axs[0].set_xlabel(fig_props["ax0_xlabel"])
                    plt.tight_layout()
                    fig.savefig(
                        fig_props["save_path"] + key + fig_props["response"] + ".pdf"
                    )
            else:
                # subfigs[i].suptitle(f'Linearized {key} Feature')
                axs = subfigs[i].subplots(
                    nrows=1, ncols=4, gridspec_kw={"width_ratios": [4.5, 2, 2, 2]}
                )
                axs[0].set_title(f"Linearized {key} Feature")
                fig, axs = self.linearization_plot(
                    key, axs=axs, fig=fig, std=std, gray_line_params=gray_line_params
                )
            # Make figure labels
            # Label the axes

            # for ax, label in zip(axs, labels[i*4:(i+1)*4]):
            #   ax.text(0.02, 0.95, label)

        # This will be written in the manuscript figure caption
        # fig.suptitle(f'Linearized Features {fig_props["response"]}')
        if not fig_props["multiple_fig"] and fig_props["save"]:
            axs[0].set_xlabel(fig_props["ax0_xlabel"])
            fig.savefig(fig_props["save_path"] + "LinerizationSummary.pdf")

        return self

    def analyze_feature(
        self,
        feat_key,
        std=False,
        plot_cv=0,
        con_thres=0.01,
        opt_gamma_method="Xv",
        opt_cv: dict = None,
        opt_dist: dict = None,
        spec_models: dict = None,
        verbose=False,
    ):
        """Function to anlayse features given certain data! (:
        The internet says (https://docs.python-guide.org/writing/gotchas/):
        Python's default arguments are evaluated once when the function is defined, not each time the function is called (like it is in say, Ruby).
        """

        # List of results, should be in the same order as the pandas dataframe.
        if opt_cv is None:
            opt_cv = {"active": False, "model": []}
        if opt_dist is None:
            opt_dist = {"active": False, "model": []}
        if spec_models is None:
            spec_models = {"models": [], "model_names": []}

        models = spec_models["models"]
        model_names = spec_models["model_names"]

        results = []
        results.append(feat_key)
        if std:
            X = self.data.X_std
            stdv = self.data.stdx
        else:
            X = self.data.X_
            stdv = None
        y = self.data.y_

        # Calculate the feature and linearized coef.
        x_hat, lin_coef_, lin_const_coef = self.regress_linearized_coeff(
            self.feat_fun_dict[feat_key], std=std
        )
        nrmse_linfeat = (
            100
            * mean_squared_error(y, X @ (lin_coef_.reshape(-1)), squared=False)
            / (np.max(y) - np.min(y))
        )

        # These if statements could be improved, for speed, but it works for now
        if "PLS" in opt_cv["model"] or "PLS" in opt_dist["model"]:
            cv_dict_pls = optimize_cv(
                X,
                y,
                max_comps=10,
                alpha_lim=[10e-5, 10e3],
                folds=5,
                nb_stds=1,
                algorithm="PLS",
                plot_components=plot_cv,
                std=std,
                stdv=stdv,
                min_distance_search=opt_dist["active"],
                featlin=lin_coef_,
                verbose=verbose,
            )
        if "RR" in opt_cv["model"] or "RR" in opt_dist["model"]:
            cv_dict_rr = optimize_cv(
                X,
                y,
                max_comps=10,
                alpha_lim=[10e-5, 10e4],
                folds=5,
                nb_stds=1,
                algorithm="RR",
                plot_components=plot_cv,
                std=std,
                stdv=stdv,
                min_distance_search=opt_dist["active"],
                featlin=lin_coef_,
                verbose=verbose,
            )

        if "PLS" in opt_cv["model"]:
            rmse_min_comp = cv_dict_pls["cv_res"]["rmse_min_param"]
            models.append(
                PLSRegression(n_components=rmse_min_comp, tol=1e-7, scale=False)
            )
            model_names.append("PLS " + str(rmse_min_comp) + " comp")

        if "PLS" in opt_dist["model"]:
            comp = cv_dict_pls["l2_distance_res"]["l2_min_param"]
            if f"PLS {comp} comp" in model_names:
                id = model_names.index(f"PLS {comp} comp")
                model_names.remove(f"PLS {comp} comp")
                print(f"popping model with {id}")
                models.pop(id)
                # models.remove(PLSRegression(n_components=comp, tol=1e-7, scale=False))
            models.append(PLSRegression(n_components=comp, tol=1e-7, scale=False))
            model_names.append(f"PLS {comp} comp")

        if "RR" in opt_cv["model"]:
            alpha_min_rmse = cv_dict_rr["cv_res"]["rmse_min_param"]
            models.append(Ridge(alpha=alpha_min_rmse))
            model_names.append("RR " + str(alpha_min_rmse))

        if "RR" in opt_dist["model"]:
            alpha = cv_dict_rr["l2_distance_res"]["l2_min_param"]
            models.append(Ridge(alpha=alpha))
            model_names.append(f"RR: {alpha:.5f}")

        model_names.append("lfun")
        self.nullspace_dict[feat_key] = dict.fromkeys(model_names)
        self.nullspace_dict[feat_key]["lfun"] = dict.fromkeys(
            ["feature_fun", "lin_coef", "nrmse"]
        )
        self.nullspace_dict[feat_key]["lfun"]["feature_fun"] = self.feat_fun_dict[
            feat_key
        ]
        self.nullspace_dict[feat_key]["lfun"]["lin_coef"] = lin_coef_
        self.nullspace_dict[feat_key]["lfun"]["nrmse"] = nrmse_linfeat
        self.nullspace_dict[feat_key]["lfun"]["x_hat"] = x_hat
        if verbose:
            print(model_names)

        for i, model in enumerate(models):
            results.append(model_names[i])
            reg = model.fit(X, y)
            nrmse_reg = (
                100
                * mean_squared_error(y, X @ (reg.coef_.reshape(-1)), squared=False)
                / (np.max(y) - np.min(y))
            )

            self.nullspace_dict[feat_key][model_names[i]] = dict.fromkeys(
                ["model", "nrmse", "nulls_label", "nulls"]
            )

            self.nullspace_dict[feat_key][model_names[i]]["model"] = copy.deepcopy(reg)
            self.nullspace_dict[feat_key][model_names[i]]["nrmse"] = nrmse_reg

            # Create Nullspace object
            # Nullspace object handles standardization itself and expect coefficeints corresponding to
            # data that is NOT standardized. Logic could be improved here.
            if std:
                reg_coef_nullsp = reg.coef_.reshape(-1) / self.data.stdx
                lin_coef_nullsp = lin_coef_.reshape(-1) / self.data.stdx
            else:
                reg_coef_nullsp = reg.coef_.reshape(-1)
                lin_coef_nullsp = lin_coef_.reshape(-1)
            # Setup the nullspace object with data that is NOT standardized
            nulls_ = Nullspace(self.data)
            # For the nullspace correction we apply std false in all cases,
            # becasue we work woth retransformed regression coefficeints.
            nulls_ = nulls_.nullspace_correction(
                w_alpha=reg_coef_nullsp,
                w_alpha_name=model_names[i],
                w_beta=lin_coef_nullsp,
                w_beta_name="",
                std=std,
                plot_results=False,
                save_plot=0,
                con_thres=con_thres,
                opt_gamma_method=opt_gamma_method,
                multi_gammas=False,
                verbose=verbose,
            )

            self.nullspace_dict[feat_key][model_names[i]]["nulls"] = nulls_
            label = format_label(
                nulls_.max_gamma, method=nulls_.opt_gamma_method, con_val=nulls_.con_val
            )
            self.nullspace_dict[feat_key][model_names[i]]["nulls_label"] = label

        return self

    def linearization_plot(
        self,
        feat_key,
        axs=None,
        fig=None,
        std=False,
        label_dict: dict = None,
        gray_line_params=[1, 100, 5],
    ):
        """Plot a row of 3 subplots.
        1: Regression coefficients
        2: Linearization Analysis
        3: Pearson correlation coefficient

        Parameters
        ----------
        feat_key : str
            Feature key
        axs : list, optional
            List of axes, by default None
        fig : [type], optional
            Figure, by default None
        std : bool, optional
            Standardize data, by default False
        label_dict : dict, optional
            Dictionary with labels, by default None
        gray_line_params : list, optional
            List of parameters for gray line, by default [1, 500, 5]
            gray_line_params[0] = factor for lowest coefficient
            gray_line_params[1] = factor for highest coefficient
            gray_line_params[2] = number of gray lines
        """

        if label_dict is None:
            label_dict = {"xlabel": "Voltage (V)"}

        if axs is None:
            fig, axs = plt.subplots(
                1, 4, gridspec_kw={"width_ratios": [4, 2.5, 2.5, 2.5]}, figsize=(23, 4.72)
            )
        x_label = label_dict["xlabel"]
        lin_coef_ = self.nullspace_dict[feat_key]["lfun"]["lin_coef"]
        # if std:
        #    lin_coef_ = lin_coef_*self.data.stdx

        # Linearized coefficients
        # nrmse_linfeat = self.nullspace_dict[feat_key]["lfun"]["nrmse"]
        axs[0].plot(
            self.data.x,
            lin_coef_.reshape(-1),
            label=r"$\beta_{T1}$: Feature coefficients",
            color="k",
        )
        # marker=self.marker_list[0], markevery=(0, 30),  markersize=9)
        # +f" NRMSE: {nrmse_linfeat:.2f}%"

        # Other regression coefficients

        keys_models = list(self.nullspace_dict[feat_key].keys())
        model_names = [
            keys_models[i] for i in range(len(keys_models)) if keys_models[i] != "lfun"
        ]
        for j, model_name in enumerate(model_names):
            reg = self.nullspace_dict[feat_key][model_name]["model"]
            # nrmse_reg = self.nullspace_dict[feat_key][model_name]["nrmse"]

            # nrmse_models = 100*mean_squared_error(self.data.X_std@(lin_coef_.reshape(-1)), self.data.X_std@(reg.coef_.reshape(-1)), squared=False)/(np.max(self.data.y_)-np.min(self.data.y_))
            # \n NRMSE(PLS, Lin): {nrmse_models}"
            if "PLS" in model_name:
                marker = "s"
                color = "#332288"
                label = (
                    model_names[j].split()[0]
                    + ", "
                    + model_names[j].split()[1]
                    + " "
                    + model_names[j].split()[2]
                    + "."
                )
            if "RR" in model_name:
                marker = "o"
                color = "#117733"
                label = rf"{model_name.split()[0][:-1]}, $\lambda=${float(model_name.split()[1]):.2f}"
            else:
                marker = "s"
            axs[0].plot(
                self.data.x,
                reg.coef_.reshape(-1),
                label=r"$\beta$:" + " " + label,
                lw=2,
                color=color,
                marker=marker,
                markevery=(5 * (j + 1), 65),
                markersize=5,
            )
            # , NRMSE: {nrmse_reg:.2f}%

            nulls_ = self.nullspace_dict[feat_key][model_name]["nulls"]
            if std:
                X = nulls_.data.X_std
                key_alpha = "w_alpha_std"
                key_beta = "w_beta_std"
            else:
                X = nulls_.data.X_
                key_alpha = "w_alpha"
                key_beta = "w_beta"
            # Plot the nullspace corrected coefficients
            # nrmse_reg_nulls = 0.00
            marker = "v"
            # get the first word from the model name
            # model_names_split = model_names[j].split()
            # model_initials = model_names_split[0]
            label = self.nullspace_dict[feat_key][model_name]["nulls_label"]
            # nrmse_reg_nulls = nrmse(self.data.y_, X@(reg.coef_.reshape(-1)+nulls_.nullsp['v_'][-1,:]))
            # , NRMSE: {nrmse_reg_nulls:.2f}%
            axs[0].plot(
                self.data.x,
                reg.coef_.reshape(-1) + nulls_.nullsp["v_"][-1, :],
                label=r"$\beta + \mathbf{v}:$"
                + " "
                + label.split()[-2]
                + " "
                + label.split()[-1],
                lw=2,
                color="#AA4499",
                marker=marker,
                markevery=(5 * (j + 4), 65),
                markersize=5,
                zorder=-1,
            )

            if gray_line_params is not None:
                # Calculate 10 differecn nullspace correction vectors
                # Get the gamma values
                gamma_vals = np.logspace(
                    np.log10(gray_line_params[0] * nulls_.max_gamma),
                    np.log10(gray_line_params[1] * nulls_.max_gamma),
                    gray_line_params[2],
                )
                # gamma_vals = np.linspace(nulls_.max_gamma, 10*nulls_.max_gamma, 5)

                v, v_, norm_, gamma = nulls_.nullspace_calc(
                    key_alpha, key_beta, X, gs=gamma_vals
                )

                for i in range(v_.shape[0]):
                    if i == 0:
                        axs[0].plot(
                            self.data.x,
                            reg.coef_.reshape(-1) + v_[i, :],
                            label=label.split()[0],
                            lw=0.4,
                            color="#2b2b2b",
                            zorder=-1,
                        )
                    else:
                        axs[0].plot(
                            self.data.x,
                            reg.coef_.reshape(-1) + v_[i, :],
                            lw=0.4,
                            color="#2b2b2b",
                            zorder=-1,
                        )

        axs[0].set_ylabel(r"$\beta$")
        axs[0].set_xlabel(x_label)
        axs[0].set_xlim([2.0, 3.5])
        label = self.nullspace_dict[feat_key][model_name]["nulls_label"]

        # label=r'close to $\mathcal{\mathbf{N}}(X) xyz$'
        # print(label)

        x = self.data.x
        if std:
            y2 = nulls_.nullsp["w_alpha_std"] + nulls_.nullsp["v_"][-1, :]
            axs[0].fill_between(
                x.reshape(-1),
                nulls_.nullsp["w_alpha_std"],
                y2=y2,
                color="darkgrey",
                zorder=-1,
                alpha=0.35,
            )  # , label=label)
        else:
            y2 = nulls_.nullsp["w_alpha"] + nulls_.nullsp["v_"][-1, :]
            axs[0].fill_between(
                x.reshape(-1),
                nulls_.nullsp["w_alpha"],
                y2=y2,
                color="darkgrey",
                zorder=-1,
                alpha=0.35,
            )  # , label=label)

        # legend_fontsize = mpl.rcParams["legend.fontsize"]
        # legend = axs[0].legend(loc="best", frameon=False, fontsize=legend_fontsize - 2)
        # legend.legendHandles[-1].set_linewidth(0.1)
        # Middle: Non-linearity check
        # How good is the linear approximation:
        X = self.data.X
        feat_nonlin = np.zeros(len(X))
        a = np.mean(X, axis=0)
        fun = self.nullspace_dict[feat_key]["lfun"]["feature_fun"]
        fun_a = fun(a)

        for j in range(len(X)):
            feat_nonlin[j] = fun(X[j, :])

        # Throw it all into a plotter
        x_hat = self.nullspace_dict[feat_key]["lfun"]["x_hat"]

        # Prediction error
        nulls_.scatter_predictions(ax=axs[1])

        plot_linearized_nonlinear_comp(
            feat_nonlin,
            x_hat,
            self.data.y_,
            fun_a,
            cmap=self.cmap,
            title="",
            xlabel="Linearized Feature",
            ylabel="Feature",
            ax=axs[2],
        )

        # Right: Pearson correlation coefficient
        plot_pearson_corr_coef_comp(
            feat_nonlin,
            self.data.y_,
            self.cmap,
            title="Person Correlation",
            xlabel="Feature",
            ylabel="y",
            ax=axs[3],
        )

        return fig, axs


def jax_moment(X, power):
    """rewriting the sample moment without prefactor! 1/n
    operating on a single row of a matrix
    using jax impolemtations to allow for autodifferentiation
    """
    X_tilde = jnp.array(X) - jnp.mean(X)
    if len(X.shape) == 2:
        shape = X.shape[1]
    else:
        shape = X.shape[0]
    return jnp.sum(jnp.power(X_tilde, power)) / shape


def plot_linearized_nonlinear_comp(
    feature_non_lin,
    feature_linearized,
    y_train,
    center_taylor,
    cmap,
    title="",
    xlabel="",
    ylabel="",
    ax=None,
):
    """Plots the linear and non linear scalar feature values and makes a visual comparison!"""
    # Get axis object in case no axis object was passed explicitly
    # if ax is None:
    #    print('ax is None, create aixs')
    #    ax = plt.gca()
    # y_train_norm = (y_train - y_train.min()) / (y_train.max() - y_train.min())

    _ = mean_squared_error(feature_non_lin, feature_linearized, squared=False) / (
        np.max(feature_non_lin) - np.min(feature_non_lin)
    )
    # ind_ = np.where((y>=350) & (y<=1500))
    # rmse_ = mean_squared_error(feature_non_lin[ind_], feature_linearized[ind_], squared=False)

    # rss = np.sum(np.abs(feature_non_lin - feature_linearized))

    for i in range(len(feature_non_lin)):
        ax.scatter(
            feature_linearized[i], feature_non_lin[i], color="k", s=120
        )  # color=cmap(y_train_norm[i]),

    _ = ax.scatter(
        center_taylor,
        center_taylor,
        marker="+",
        s=35**2,
        linewidths=3,
        label=r"$\mathbf{a}=\overline{\mathbf{x}}$",
    )
    vals = np.linspace(feature_linearized.min(), feature_linearized.max(), 10)

    ax.plot(vals, vals)

    # cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)

    # How good is the linear approximation:
    # mean_fx = np.mean(feature_non_lin)
    # f_mean_x = center_taylor
    # rel_deviation = np.abs((f_mean_x-mean_fx)/mean_fx)

    # textstr = '\n'.join((
    #    'NRMSE: %.3f' % (nrmse),
    # 'RMSE Central Region: %.2f' % (rmse_),
    # 'Dev at center: %.2f' % (100*rel_deviation) + '%',
    #    ))
    # h2 = ax.plot([], [], ' ', label=textstr)

    # Fix overlapping axis ticks in case of small numbers
    if np.abs(feature_linearized.max() - feature_linearized.min()) < 0.01:
        ax.ticklabel_format(axis="both", style="sci", useOffset=True, scilimits=(0, 0))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # handles, labels = ax.get_legend_handles_labels()
    # order = [1, 0]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.legend(loc="best", frameon=False)
    # ax.grid()

    return


def plot_pearson_corr_coef_comp(
    feature_non_lin,
    y_train,
    cmap,
    title="Person Correlation",
    xlabel="",
    ylabel="",
    ax=None,
):
    """Plots the feature values and y response values, display the person corr coeff"""
    # Get axis object in case no axis object was passed explicitly
    # if ax is None:
    #    print('ax is None, create aixs')
    #    ax = plt.gca()

    corr_coeff = np.corrcoef(np.column_stack((feature_non_lin, y_train)), rowvar=False)

    # y_train_norm = (y_train - y_train.min()) / (y_train.max() - y_train.min())
    for i in range(len(feature_non_lin)):
        ax.scatter(
            feature_non_lin[i], y_train[i], s=120, color="k"
        )  # color=cmap(y_train_norm[i]), s=100)

    # cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)

    h2 = ax.plot([], [], " ", label=rf"$\rho=${corr_coeff[0, 1]:.2f}")  # noqa
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # handles, labels = ax.get_legend_handles_labels()
    # order = [1,0]
    ax.legend(loc="best", frameon=False)
    # ax.grid()

    return
