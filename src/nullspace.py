# Copyright: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
from __future__ import annotations
import numpy as np
from numpy import linalg as LA
from scipy import linalg  # type: ignore
from scipy.optimize import minimize  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import matplotlib.pylab as plt  # type: ignore
import time
from typing import Union
from typing import Protocol
import warnings
from src.plotting_utils import plot_nullspace_analysis
from src.plotting_utils import scatter_predictions as scatter_predictions_helper
from src.hd_data import HD_Data


# source: https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None):
        ...

    def predict(self, X):
        ...

    def score(self, X, y, sample_weight=None):
        ...

    def set_params(self, **params):
        ...


class Nullspace:
    """Methods to calulate the nullspace correction
    The nullspace correction is applied to compare the two regression coefficients w_alpha and w_beta.
    """

    def __init__(self, data, **kwargs):
        self.weights: dict = {}
        self.nullsp: dict = {}
        self.data: HD_Data = data
        self.con_thres: float = None
        self.con_val: float = None
        self.opt_gamma_method: str = None
        self.max_gamma: float = None
        self.label_dict: dict = None
        # Active data set
        self.X = self.data.X_
        self.d = self.data.d
        self.y = self.data.y_
        self.std = False

    def set_standardization(self, std: bool = False) -> None:
        """Set whether the active data set is standardized or not."""
        if std:
            self.X = self.data.X_std
            self.std = True
        else:
            self.X = self.data.X_
            self.std = False
        # Perform this calcualtions only once!
        self.XtX = self.X.T @ self.X

    def learn_weights(self, models: list[ScikitModel], string_id: str) -> Nullspace:
        """Learn weights from data and sotre them in the dictionary.
        Always learns for the standardized and non-standardized data, X."""
        for i, model in enumerate(models):
            reg = model.fit(self.data.X_, self.data.y_)
            coef = reg.coef_.reshape(-1)
            self.weights[string_id[i]] = coef

            # For some synthethic data std might be 0 -> Standardization not possible!
            try:
                reg_std = model.fit(self.data.X_std, self.data.y_)
                coef_std = reg_std.coef_.reshape(-1)
                self.weights[string_id[i] + " std"] = coef_std
                self.weights[string_id[i] + " std retrans"] = coef_std / self.data.stdx
            except ZeroDivisionError:
                self.weights[string_id[i] + " std"] = "Undefined Std = 0 for some column"
                self.weights[
                    string_id[i] + " std retrans"
                ] = "Undefined Std = 0 for some column"
        return self

    def set_nullspace_weights(
        self,
        *,
        w_alpha: np.ndarray = None,
        key_alpha: np.ndarray = None,
        w_alpha_name: str = None,
        w_beta: np.ndarray = None,
        key_beta: np.ndarray = None,
        w_beta_name: str = None,
    ) -> None:
        if (w_alpha is None) & (key_alpha is None):
            NameError(
                "Either w_alpha is passed directly or key for learned coefficients must be passed"
            )

        if (w_beta is None) & (key_beta is None):
            NameError(
                "Either w_alpha is passed directly or key for learned coefficients must be passed"
            )
        # Names to be used in the plot legend
        self.w_alpha_name = w_alpha_name
        self.w_beta_name = w_beta_name

        if w_alpha is None:
            self.nullsp["w_alpha"] = self.weights[key_alpha]
            self.nullsp["w_alpha_std"] = self.weights[key_alpha + " std"]
        else:
            self.nullsp["w_alpha"] = w_alpha
            self.nullsp["w_alpha_std"] = w_alpha * self.data.stdx

        if w_beta is None:
            self.nullsp["w_beta"] = self.weights[key_beta]
            self.nullsp["w_beta_std"] = self.weights[key_beta + " std"]
        else:
            self.nullsp["w_beta"] = w_beta
            self.nullsp["w_beta_std"] = w_beta * self.data.stdx

        if self.std:
            self.w_alpha = self.nullsp["w_alpha_std"]
            self.w_beta = self.nullsp["w_beta_std"]
        else:
            self.w_alpha = self.nullsp["w_alpha"]
            self.w_beta = self.nullsp["w_beta"]
        self.w = self.w_alpha - self.w_beta

    def nullspace_analysis(
        self,
        *,
        nullspace_path: bool = True,
        con_thres: float = 0.5,
        opt_gamma_method: str = "Xv",
        analyse_objective_trajectory: bool = False,
        plot_results: bool = False,
        save_plot: bool = False,
        path_save: str = "",
        file_name: str = "",
        **kwargs,
    ) -> Union[Nullspace, tuple[Nullspace, plt.figure, plt.axes]]:
        """Run the nullspace analysis. Make sure to learn/set the relevant weights first."""

        self.opt_gamma_method = opt_gamma_method
        self.con_thres = con_thres
        range_y = np.max(self.y) - np.min(self.y)

        if "gamma_vals" in kwargs.keys():
            self.max_gamma = np.max(kwargs["gamma_vals"])
            # drop the key from the dictionary
            kwargs.pop("gamma_vals")
            cons_dict = self.eval_constraint(
                self.max_gamma, methods=self.opt_gamma_method
            )
            self.con_val = cons_dict[self.opt_gamma_method]
            print(f"Constraint value: {self.con_val:.12f}")
            (
                self.nullsp["v_"],
                self.nullsp["norm_"],
                self.nullsp["gamma"],
            ) = self.nullspace_calc(gs=self.max_gamma)
        else:
            # If the value of the constraint is between 0 and -1 use percentage ratio
            if opt_gamma_method == "NRMSE" and self.con_thres < 0:
                nrmse_alpha = (
                    100
                    * mean_squared_error(self.y, self.X @ (self.w_alpha), squared=False)
                    / range_y
                )
                nrmse_beta = (
                    100
                    * mean_squared_error(self.y, self.X @ (self.w_beta), squared=False)
                    / range_y
                )
                self.con_thres = np.abs(self.con_thres) * np.abs(nrmse_alpha - nrmse_beta)
                print("NRMSE constraint threshold: ", self.con_thres)

            self.optimize_gamma(nullspace_path=nullspace_path)

        if plot_results:
            fig, ax = self.plot_nullspace_analysis(**kwargs)
            if save_plot:
                fig.savefig(path_save + file_name)

        if analyse_objective_trajectory:
            # We have to be far away enough form the numeric precision to make this work.
            if self.con_val > 1e-12:
                self.objective_function_trajectory()
            else:
                print(
                    f"Constraint value: {self.con_val}, is too close to the numeric precision for path analysis"
                )
        if plot_results:
            return self, fig, ax
        else:
            return self

    def objective_function_trajectory(
        self,
        *,
        gamma_vals: np.ndarray = None,
    ):
        """Try an array of gamma values to see how the proposed NRMSE objective changes."""
        if gamma_vals is None:
            gamma_vals = np.logspace(5, -5, 40)

        (
            self.nullsp["v_"],
            self.nullsp["norm_"],
            self.nullsp["gamma"],
        ) = self.nullspace_calc(gs=gamma_vals)

        nrmse_list = []
        xv = []
        for gamma in gamma_vals:
            # Evaluate the NRMSE metric
            cons_dict = self.eval_constraint(gamma=gamma, methods=["NRMSE", "Xv"])
            nrmse_list.append(cons_dict["NRMSE"])
            xv.append(cons_dict["Xv"])

        # MAke a plot with two y-axis to show the NRMSE and the Xv metric
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(gamma_vals, nrmse_list, label=r"$\Delta$ NRMSE")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\Delta$ NRMSE")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.plot(gamma_vals, xv, label=r"$\tilde{n}$", color="red")
        ax2.set_ylabel(r"$\tilde{n}$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax.legend(loc="lower left")
        ax2.legend(loc="upper right")
        # fig.savefig("NRMSE_Xv.pdf")
        fig.show()

    def optimize_gamma(
        self,
        nullspace_path: bool = False,
    ) -> None:
        """Optimize the gamma parameter for the nullspace correction."""
        # gamma, con_val = self.scipy_opt_gamma(verbose=False)
        gamma, con_val = self.naive_opt_gamma(verbose=False)
        self.max_gamma = gamma
        self.con_val = con_val

        if nullspace_path & (type(self.max_gamma) in [np.float64, float, np.float32]):
            gamma_vals = np.geomspace(10 ** (-12), self.max_gamma + 2 * (10 ** (-12)), 30)
        else:
            gamma_vals = np.array(self.max_gamma).reshape(1)

        (
            self.nullsp["v_"],
            self.nullsp["norm_"],
            self.nullsp["gamma"],
        ) = self.nullspace_calc(gs=gamma_vals)
        cons_dict = self.eval_constraint(gamma_vals[-1], methods=self.opt_gamma_method)
        con_val = cons_dict[self.opt_gamma_method]
        print(f"Constraint value: {con_val:.12f}, Method {self.opt_gamma_method}")

        if 0:
            self.nullsp["v"] = np.array(
                [self.nullsp[self.key_beta] - self.nullsp[self.key_alpha]]
            )
            self.nullsp["v_"] = np.array(
                self.nullsp[self.key_beta] - [self.nullsp[self.key_alpha]]
            )
            self.nullsp["gamma"] = 0
            self.max_gamma = np.inf

    def naive_opt_gamma(
        self,
        gammas_inital: np.ndarray = np.geomspace(10**7, 10 ** (-7), 10),
        verbose: bool = True,
    ):
        """This is a simple way of optimizing gamma, for experimentation purposes only."""
        depth = 30
        eps = 10**-7
        cons_list = []
        gammas_ = gammas_inital
        if gammas_.ndim == 0:
            gammas_ = gammas_.reshape(1)

        tick = time.time()
        for i in range(depth):
            for j, gamma in enumerate(gammas_):
                cons_dict = self.eval_constraint(gamma, methods=[self.opt_gamma_method])
                con_val = cons_dict[self.opt_gamma_method].copy()
                cons_list.append(con_val)
                if con_val >= eps + self.con_thres:
                    break

            if con_val <= (eps + self.con_thres) and con_val >= (eps + self.con_thres):
                break

            else:
                if i == 0:
                    gammas = np.geomspace(gammas_inital[j - 1], gammas_inital[j], 3)
                    # Testing gamma in the middle of the interval to half the interval
                    gammas_ = np.array(gammas[1]).reshape(1)
                else:
                    if con_val >= eps + self.con_thres:
                        idx_high = 0
                    else:
                        idx_high = 1
                    gammas = np.geomspace(gammas[idx_high], gammas[idx_high + 1], 3)
                    gammas_ = np.array(gammas[1]).reshape(1)
        tock = time.time()

        if verbose:
            print(f"Optimization took {tock-tick:.2f} seconds")
            print(f"Optimization depth: {i}, max depth: {depth}")
            print(
                f"Gamma value corresponding to {np.abs(self.con_thres):.1e} % is {gamma}"
            )
            print(f"Constraint value: {con_val:.3f}")
        return gamma, con_val

    def scipy_opt_gamma(self, verbose: bool = True) -> float:
        tick = time.time()
        if self.opt_gamma_method == "Xv":
            warnings.warn(
                "Warning: The function scipy optimize gamma is not fully tested yet for Xv cosntraints!"
            )
            raise NotImplementedError(
                "The function optimize gamma is not fully tested yet for Xv constraints, you have to remove this line for testing it."
            )

        def constraint(x):
            con = self.eval_constraint(x, methods=[self.opt_gamma_method])[
                self.opt_gamma_method
            ]
            return self.con_thres - con

        def objective_gamma(x):
            return x

        solution = minimize(
            objective_gamma,
            50,
            method="SLSQP",
            bounds=[(10**-7, 10**7)],
            constraints={"type": "ineq", "fun": constraint},
            tol=10**-7,
        )
        tock = time.time()
        if verbose:
            print(solution.x[0])
            print(f"Scipy Optimization took {tock-tick:.2f} seconds")
        return solution.x[0], constraint(solution.x[0])

    def eval_constraint(
        self,
        gamma: float,
        methods: str = "NRMSE",
    ) -> list[float]:
        """Evaluate the constraint associated with the constraint method and gamma value."""
        v_, _, _ = self.nullspace_calc(gs=np.array(gamma))
        span_y = np.max(self.y) - np.min(self.y)
        y_pred_alpha = self.X @ self.w_alpha
        y_pred_alpha_v_ = self.X @ (self.w_alpha + v_.reshape(-1))
        constraint_vals = {}
        if "MSE" in methods:
            mse_reg = mean_squared_error(self.y, y_pred_alpha, squared=False)
            mse_nulls = mean_squared_error(self.y, y_pred_alpha_v_, squared=False)
            val = np.abs(mse_reg - mse_nulls)
            constraint_vals["MSE"] = val
        if "NRMSE" in methods:
            nrmse_reg = (
                100 * mean_squared_error(self.y, y_pred_alpha, squared=False) / span_y
            )
            nrmse_nulls = (
                100 * mean_squared_error(self.y, y_pred_alpha_v_, squared=False) / span_y
            )
            val = np.abs(nrmse_reg - nrmse_nulls) + 100 * np.sqrt(
                self.X.shape[1]
            ) * np.average(self.X @ v_.reshape(-1) ** 2) / (span_y)
            constraint_vals["NRMSE"] = val
        if "Xv" in methods:
            val = (
                100
                * np.sqrt(np.average((self.X @ v_.reshape(-1)) ** 2))
                / (np.max(y_pred_alpha) - np.min(y_pred_alpha))
            )
            constraint_vals["Xv"] = val
        return constraint_vals

    def nullspace_calc(
        self, gs: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This functions performs a nulspace normalization of regression coefficents.
        The problem is set up such that the L2 norm differences between ther regression
        coefficients is minimzed.

        Parameters
        ----------
        gs : ndarray, default=None
            1D array of penalizations of deviations form the nulspace vector

        Returns
        ----------
        v_ : ndarray
            2D array length(gs) times length w_alpha
            Regression coefficeints minimizing L2 norm but deviations from the nullsapce according gs are allowed
        norm_ : ndarray
            1D array of length(gs) storing the L2 norm between the modified regression coefficient and remaining regression coefficeint
        gs : ndarray
            1D array of penalizations of deviations form the nulspace vector
            Either equal to inpiuts or derived in the function
        """

        if gs is None:
            gs = np.geomspace(
                10 ** (-5), (10**4) * (np.max(self.X) - np.min(self.X)), 30
            )

        # Build helper vectors and matrices, not optimized for speed or memory usage!
        shape = self.X.shape
        I_ = np.identity(shape[1])

        nb_gs = gs.size
        if gs.ndim == 0:
            gs = gs.reshape(1)
        v_ = np.zeros((nb_gs, shape[1]))
        norm_ = np.zeros(nb_gs)
        for i, g in enumerate(gs):
            v_[i, :] = -linalg.inv(g * self.XtX + I_) @ self.w
            norm_[i] = LA.norm(self.w + v_[i, :], 2)

        # SVD decomposition of X for the original optimization problem
        # U, s_, Vh = linalg.svd(X)
        # s = np.zeros((shape[0], shape[1]))
        # sst = np.zeros((shape[0], shape[0]))
        # st = np.zeros((shape[1], shape[0]))

        # np.fill_diagonal(sst, (s_*s_))
        # np.fill_diagonal(s, s_)
        # np.fill_diagonal(st, s_)

        # Approach 1: Full equations
        # S = st@ssti@s
        # v = (Vh.T@S@Vh - I)@w

        # Working with block matrices.
        # This doesnt solve the issue of poor conditioned X....
        # if comp_block:
        # v11 = Vh.T[:shape[0],:shape[0]]
        # v12 = Vh.T[:shape[0],shape[0]:]
        # v21 = Vh.T[shape[0]:,:shape[0]]
        # v22 = Vh.T[shape[0]:,shape[0]:]
        # left = np.concatenate((-v12@v12.T, -v22@v12.T), axis=0)
        # right = np.concatenate((-v12@v22.T, -v22@v22.T), axis=0)
        # v_ = np.concatenate((left, right), axis=1)@w

        return v_, norm_, gs

    def plot_nullspace_analysis(
        self,
        *,
        title: str = "",
        **kwargs,
    ) -> tuple[plt.figure, plt.axes]:
        self.generate_nullspace_labels()
        if "ax_labelstr" not in kwargs.keys():
            kwargs["ax_labelstr"] = ("a)", "b)")
        fig, ax = plot_nullspace_analysis(
            w_alpha=self.w_alpha,
            w_beta=self.w_beta,
            v=self.nullsp["v_"],
            X=self.X,
            d=self.d,
            y=self.y,
            name=title,
            label_dict=self.label_dict,
            **kwargs,
        )
        return fig, ax

    def generate_nullspace_labels(self):
        nrmse_alpha = (
            100
            * mean_squared_error(self.y, self.X @ (self.w_alpha), squared=False)
            / (np.max(self.y) - np.min(self.y))
        )
        nrmse_alpha_v = (
            100
            * mean_squared_error(
                self.y, self.X @ (self.w_alpha + self.nullsp["v_"][-1, :]), squared=False
            )
            / (np.max(self.y) - np.min(self.y))
        )
        nrmse_beta = (
            100
            * mean_squared_error(self.y, self.X @ (self.w_beta), squared=False)
            / (np.max(self.y) - np.min(self.y))
        )
        self.label_dict = {}
        fprecision = 3
        nrmse_alpha_str = f"NRMSE: {nrmse_alpha:.{fprecision}f}%"
        nrmse_beta_str = f"NRMSE: {nrmse_beta:.{fprecision}f}%"
        self.label_dict["alpha"] = f"{self.w_alpha_name}, " + nrmse_alpha_str
        self.label_dict["alpha pred"] = (
            r"$\mathbf{X}$" + f"{self.w_alpha_name}, " + nrmse_alpha_str
        )
        self.label_dict["beta"] = f"{self.w_beta_name}, " + nrmse_beta_str
        self.label_dict["beta pred"] = (
            r"$\mathbf{X}$" + f"{self.w_beta_name}, " + nrmse_beta_str
        )
        text_pred = r"$\mathbf{X}($" + self.w_alpha_name + r"$+\mathbf{v}_\gamma)$"
        text_coef = self.w_alpha_name + r"$+\mathbf{v}_\gamma$"
        gamma_str = (
            r"$\gamma\approx$"
            + f"{dec_sci_switch(self.max_gamma, decimal_switch=1, sci_acc=1)}"
        )
        # text_nullsp = (
        #    # r"$\in\mathcal{\mathcal{\widetilde{N}}}(\mathbf{X})$ , $\gamma\approx$"
        #    ", "
        #    + gamma_str
        #    + ", "
        #    + r"$\Delta_{NRMSE}:$ "
        #    + f"{self.con_val:.{fprecision}f}%"
        # )
        text_nullsp = (
            ", " + gamma_str + ", " + r"$NRMSE:$ " + f"{nrmse_alpha_v:.{fprecision}f}%"
        )
        self.label_dict["alpha+v pred"] = text_pred + text_nullsp
        self.label_dict["alpha+v coef"] = text_coef + text_nullsp
        self.label_dict["Xv"] = r"$\mathbf{Xv}_\gamma, $" + gamma_str

    def scatter_predictions(
        self,
        *,
        ax: bool = None,
        return_fig: bool = False,
        ax_labelstr: str = "c)",
        title: str = "",
        **kwargs,
    ):
        """Method that scatters based on nullspace correction."""
        w = [(self.w_alpha + self.nullsp["v_"][-1, :]), self.w_alpha, self.w_beta]
        if self.label_dict is None:
            self.generate_nullspace_labels()

        return scatter_predictions_helper(
            self.X,
            self.y,
            np.mean(self.data.y),
            w,
            self.label_dict,
            ax=ax,
            return_fig=return_fig,
            ax_labelstr=ax_labelstr,
            title=title,
            **kwargs,
        )


def dec_sci_switch(number: float, decimal_switch: int = 3, sci_acc: int = 2) -> str:
    """Switch between decimal and scientific notation"""
    if number < 10 ** (-decimal_switch):
        return f"{number:.{sci_acc}e}"
    elif number > 1000:
        return f"{number:.2e}"
    else:
        return f"{number:.{decimal_switch}f}"
