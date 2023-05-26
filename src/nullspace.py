import numpy as np
from numpy import linalg as LA
from scipy import linalg
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pylab as plt
from src.utils import nrmse
from src.plotting_utils import plot_nullspace_correction, scatter_predictions


class Nullspace:
    """Methods to calulate the nullspace correction
    The nullspace correction is calculated between the two models w_alpha and w_beta.
    A certain NRMSE of the prediction of the model is allowed in the nullspace correction.
    """

    def __init__(self, data, **kwargs):
        self.weights = (
            {}
        )  # Dictionary to store weights learned from mean subtracted data X_
        self.nullsp = {}
        self.data = data  # Should be a BasicsCalss object or duck type
        self.con_thres = None
        self.con_val = None
        self.opt_gamma_method = None
        self.max_gamma = None

    def learn_weights(self, models, string_id):
        """Learn weights from data and sotre them in the dictionary.

        Parameters
        ----------
        model : object
            sklearn model object with methods fit(), predict() (e.g. sklearn)
            for custom models you might want to write a wrapper
        string_id : str
            model description string
        """
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
                ] = "Undefined Std = 0 ofr some column"
        return self

    def nullspace_correction(
        self,
        w_alpha=None,
        w_alpha_name=None,
        w_beta=None,
        w_beta_name=None,
        std=False,
        plot_results=False,
        save_plot=False,
        path_save="",
        file_name="",
        multi_gammas=True,
        con_thres=0.5,
        opt_gamma_method="Xv",
        gammas_inital=np.geomspace(10**11, 10 ** (-5), 10),
        **kwargs,
    ):
        """Function that calls 'nullspace_calc allowing to shorten syntax.

        Parameters
        ----------
        w_alpha : ndarray
            Regression coefficients obtained with method alpha
        w_beta : ndarray
            Regression coefficients obtained with method beta
        data_ml_obj : obj
            Object from SynMLData class
        std : bool, default=False
            Indicate whether Standardization(Z-Scoring) shall be performed.
            If yes, use X_std of the data_ml_object, if not use X.
        max_nrmse : int, default 1
            Tolerable nrmse error, to be considered ``close'' to the nullspace.
            Set this parameter with care, based on how much deviation from the
            nullspace you want to tolerate as ``insignificant''
            -1: Take np.abs(max_nrmse) times the difference between the prediction errors
        plot_results : bool, default=False
            Indicate whether to plot results
        save_results : bool, default=False
            indicate whether to save plot as pdf
        path_save : str, default=''
            path for the case save resutls is true
        fig_name : str, defautl=''
            figure file name

        Returns
        -------
        self

        depending on whether plot_results is true
        fig : object
            matplotlib figure object
        ax : object
            matplotlib axis objects
        """

        if (w_alpha is None) & ("key_alpha" not in kwargs):
            NameError(
                "Either w_alpha is passed directly or key for learned coefficients must be passed"
            )

        if (w_beta is None) & ("key_beta" not in kwargs):
            NameError(
                "Either w_alpha is passed directly or key for learned coefficients must be passed"
            )

        self.nullsp["w_beta_name"] = w_beta_name
        self.nullsp["w_alpha_name"] = w_alpha_name

        # To keep things simple, standardized weights also included here!
        # In case std==False this is not efficient, because the standardized coef. aren't used.
        if w_alpha is None:
            self.nullsp["w_alpha"] = self.weights[kwargs.get("key_alpha")]
            self.nullsp["w_alpha_std"] = self.weights[kwargs.get("key_alpha") + " std"]
        else:
            self.nullsp["w_alpha"] = w_alpha
            self.nullsp["w_alpha_std"] = w_alpha * self.data.stdx

        if w_beta is None:
            self.nullsp["w_beta"] = self.weights[kwargs.get("key_beta")]
            self.nullsp["w_beta_std"] = self.weights[kwargs.get("key_beta") + " std"]
        else:
            self.nullsp["w_beta"] = w_beta
            self.nullsp["w_beta_std"] = w_beta * self.data.stdx

        if std:
            X = self.data.X_std
            self.nullsp["w_alpha_name"] += " std"
            self.nullsp["w_beta_name"] += " std"
            key_alpha = "w_alpha_std"
            key_beta = "w_beta_std"
        else:
            X = self.data.X_
            key_alpha = "w_alpha"
            key_beta = "w_beta"

        # x = self.data.x
        self.nullsp["info"] = ""

        self.opt_gamma_method = opt_gamma_method
        self.con_thres = con_thres
        y_ = self.data.y_

        if opt_gamma_method == "NRMSE" and self.con_thres < 0:
            nrmse_alpha = (
                100
                * mean_squared_error(y_, X @ (self.nullsp[key_alpha]), squared=False)
                / (np.max(y_) - np.min(y_))
            )
            nrmse_beta = (
                100
                * mean_squared_error(y_, X @ (self.nullsp[key_beta]), squared=False)
                / (np.max(y_) - np.min(y_))
            )
            self.con_thres = np.abs(self.con_thres) * np.abs(nrmse_alpha - nrmse_beta)
            print("NRMSE constraint threshold: ", self.con_thres)

            # mse_alpha = mean_squared_error(y_, X@(self.nullsp[key_alpha]), squared=False)
            # mse_beta = mean_squared_error(y_, X@(self.nullsp[key_beta]), squared=False)
            # self.con_thres = np.abs(self.con_thres) * np.abs(mse_alpha-mse_beta)

        # Activate for debugging purposes/get more insights into how the constraints work.
        if 1:
            # Run a loop over different values of gamma to see how the proposer NRMSE metric would change.
            gamma_vals = np.logspace(5, -5, 80)
            (
                self.nullsp["v"],
                self.nullsp["v_"],
                self.nullsp["norm_"],
                self.nullsp["gamma"],
            ) = self.nullspace_calc(key_alpha, key_beta, X, gs=gamma_vals)
            nrmse_list = []
            xv = []
            for i, gamma in enumerate(gamma_vals):
                # Evaluate the NRMSE metric
                cons, con_xv = self.eval_constraint(
                    X, y_, key_alpha, key_beta, gamma, method="NRMSE_Xv"
                )
                nrmse_list.append(cons)
                xv.append(con_xv)

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
            plt.savefig("NRMSE_Xv.pdf")
            plt.show()

        self.optimize_gamma(
            gammas_inital=gammas_inital,
            X=X,
            y_=y_,
            key_alpha=key_alpha,
            key_beta=key_beta,
            multi_gammas=multi_gammas,
        )

        if plot_results:
            fig, ax = self.plot_nullspace_correction(std=std)
            if save_plot:
                fig.savefig(path_save + file_name)
            return self, fig, ax
        else:
            return self

    def optimize_gamma(
        self,
        gammas_inital=np.geomspace(10**11, 10 ** (-5), 10),
        X=None,
        y_=None,
        key_alpha=None,
        key_beta=None,
        multi_gammas=False,
    ):
        """Optimize the gamma parameter for the nullspace correction.
        This is a simple (but ineffective way) way of optimizing which should suffice for this issue.

        """

        depth = 20
        eps = 10**-7
        cons_list = []
        gammas_ = gammas_inital

        for i in range(depth):
            # print('Iteration: ', i)
            for j, gamma in enumerate(gammas_):
                con_val = self.eval_constraint(
                    X, y_, key_alpha, key_beta, gamma, method=self.opt_gamma_method
                )
                # cons = self.eval_constraint(X, y_, key_alpha, key_beta, gamma, method='NRMSE')
                cons_list.append(con_val)
                if con_val >= eps + self.con_thres:
                    break
            if con_val <= eps + self.con_thres and con_val >= eps + self.con_thres:
                break
            else:
                if i == 0:
                    gammas = np.geomspace(gammas_inital[j - 1], gammas_inital[j], 3)
                    # We will only test gamm in the middle of the interval.
                    gammas_ = [gammas[1]]
                else:
                    # TODO: Fix this, then fix nullspace paper.
                    if con_val >= eps + self.con_thres:
                        idx_high = 0
                    else:
                        idx_high = 1
                    gammas = np.geomspace(gammas[idx_high], gammas[idx_high + 1], 3)
                    gammas_ = [gammas[1]]

        self.max_gamma = gamma
        self.con_val = con_val

        # print(f'Gamma value corresponding to nrmse={np.abs(self.max_nrmse):.1e} % is {self.max_gamma:.3f}')
        # Print constraint con_xy value
        # print(f'Constraint value: {con_xv:.3f}')

        # con = {'type': 'eq', 'fun': constraint}
        # solution = minimize(objective_gamma, 5000, method='SLSQP',\
        #        bounds=[(1, 10**10)], constraints=con)
        # print(solution.x[0])
        # print(f'Contraint {constraint(solution.x[0])}')
        # self.max_gamma = solution.x[0]
        # y_range = np.max(y_) - np.min(y_)
        # gs_inital = 100*y_range
        # Find value for gamma that
        # import scipy as sp
        # gamma_upper_limit = sp.optimize.minimize(
        #     find_gamma_, 100, args=(self.nullsp[key_alpha], self.nullsp[key_beta], X, x, y_, max_nrmse),
        #     method='Nelder-Mead', bounds=[(1, 10**10)], options={'xatol' : 0.01})
        # self.max_gamma = gamma_upper_limit.x[0]

        if multi_gammas:
            gamma_vals = np.geomspace(10 ** (-12), self.max_gamma + 2 * (10 ** (-12)), 30)
        else:
            gamma_vals = [self.max_gamma]

        (
            self.nullsp["v"],
            self.nullsp["v_"],
            self.nullsp["norm_"],
            self.nullsp["gamma"],
        ) = self.nullspace_calc(key_alpha, key_beta, X, gs=gamma_vals)
        con_val = self.eval_constraint(
            X, y_, key_alpha, key_beta, gamma_vals[-1], method=self.opt_gamma_method
        )
        print(f"Constraint value: {con_val:.12f}, Method {self.opt_gamma_method}")

        if 0:
            self.nullsp["v"] = np.array([self.nullsp[key_beta] - self.nullsp[key_alpha]])
            self.nullsp["v_"] = np.array(self.nullsp[key_beta] - [self.nullsp[key_alpha]])
            self.nullsp["gamma"] = 0
            self.max_gamma = np.inf

        return

    def eval_constraint(self, X, y_, key_alpha, key_beta, gamma, method="NRMSE"):
        # print(f'Gamma: {gamma}')
        # n = self.data.X.shape[0]
        v, v_, norm_, gs = self.nullspace_calc(key_alpha, key_beta, X, gs=[gamma])
        # val = mean_squared_error(y_, X@(v_.reshape(-1)), squared=False)
        if method == "MSE":
            mse_reg = mean_squared_error(y_, X @ (self.nullsp[key_alpha]), squared=False)
            mse_nulls = mean_squared_error(
                y_, X @ (self.nullsp[key_alpha] + v_.reshape(-1)), squared=False
            )
            val = np.abs(mse_reg - mse_nulls)
        elif method == "NRMSE":
            nrmse_reg = (
                100
                * mean_squared_error(y_, X @ (self.nullsp[key_alpha]), squared=False)
                / (np.max(y_) - np.min(y_))
            )
            nrmse_nulls = (
                100
                * mean_squared_error(
                    y_, X @ (self.nullsp[key_alpha] + v_.reshape(-1)), squared=False
                )
                / (np.max(y_) - np.min(y_))
            )
            val = np.abs(nrmse_reg - nrmse_nulls) + 100 * np.sqrt(
                X.shape[1]
            ) * np.average(X @ v_.reshape(-1) ** 2) / (np.max(y_) - np.min(y_))
            # print(f'Delta MSE of gamma: {val}')
        elif method == "NRMSE_Xv":
            nrmse_reg = (
                100
                * mean_squared_error(y_, X @ (self.nullsp[key_alpha]), squared=False)
                / (np.max(y_) - np.min(y_))
            )
            nrmse_nulls = (
                100
                * mean_squared_error(
                    y_, X @ (self.nullsp[key_alpha] + v_.reshape(-1)), squared=False
                )
                / (np.max(y_) - np.min(y_))
            )
            val = np.abs(nrmse_reg - nrmse_nulls) + 100 * np.sqrt(
                X.shape[1]
            ) * np.average(X @ v_.reshape(-1) ** 2) / (np.max(y_) - np.min(y_))

            # pred_lin = X @ (self.nullsp[key_beta])
            pred_model = X @ (self.nullsp[key_alpha])
            con_xv = (
                100
                * np.sqrt(np.average((X @ v_.reshape(-1)) ** 2))
                / (np.max(pred_model) - np.min(pred_model))
            )
            # con_xv = np.sqrt(1000)*np.sum((X@v_.reshape(-1))**2)/(np.max(pred_model)-np.min(pred_model))
            # print(val)
            return [val, con_xv]
        elif method == "Xv":
            # pred_lin = X @ (self.nullsp[key_beta])
            pred_model = X @ (self.nullsp[key_alpha])
            val = (
                100
                * np.sqrt(np.average((X @ v_.reshape(-1)) ** 2))
                / (np.max(pred_model) - np.min(pred_model))
            )
        return val

    def nullspace_calc(self, key_alpha, key_beta, X, gs: np.array = None):
        """This functions performs a nulspace normalization of regression coefficents.
        The problem is set up such that the L2 norm differences between ther regression
        coefficients is minimzed.

        Parameters
        ----------
        w_alpha : ndarray
            1D array of linear feature coefficient vector
        w_beta : ndarray
            1D array of regression coefficients
        X : ndarray
            2D Data matrix that was used for estimating the regression coefficeint
            Predictions can be made via np.dot(X, w_alpha) or X@w_alpha
        x : ndarray
            1D array of values of the smooth domain
        gs : ndarray, default=None
            1D array of penalizations of deviations form the nulspace vector

        Returns
        ----------
        v : ndarray
            1D array of of regression coefficeint contianed in teh nullspace that minimize the L2-norm
        v_ : ndarray
            2D array length(gs) times length w_alpha
            Regression coefficeints minimizing L2 norm but deviations from the nullsapce according gs are allowed
        norm_ : ndarray
            1D array of length(gs) storing the L2 norm between the modified regression coefficient and remaining regression coefficeint
        gs : ndarray
            1D array of penalizations of deviations form the nulspace vector
            Either equal to inpiuts or derived in the function
        """
        w_alpha = self.nullsp[key_alpha]
        w_beta = self.nullsp[key_beta]
        if gs is None:
            gs = np.geomspace(10 ** (-5), (10**4) * (np.max(X) - np.min(X)), 30)
        # difference between coefficients
        w = w_alpha - w_beta

        # Build helper vectors and matrices, not optimized for speed or memory usage!
        shape = X.shape
        I_ = np.identity(shape[1])

        # Do the magic:
        nb_gs = len(gs)
        v_ = np.zeros((nb_gs, shape[1]))
        norm_ = np.zeros(nb_gs)

        for i, g in enumerate(gs):
            v_[i, :] = -linalg.inv(g * X.T @ X + I_) @ w
            norm_[i] = LA.norm(w_alpha + v_[i, :] - w_beta, 2)

        # SVD decomposition of X
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

        return np.inf, v_, norm_, gs

    def plot_nullspace_correction(self, std=False, title=""):
        """Method that calls plot_nullspace_correction, uses basis object.

        Parameters
        ----------
        title : str
            Suptitle of the resulting figure
        std : bool, default=False
            Indicate whether Standardization(Z-Scoring) shall be performed.
            If yes, use X_std of the data_ml_object, if not use X.

        Returns
        -------
        fig : object
            matplotlib figure object
        ax : object
            matplotlib axis objects
        """
        if std:
            X = self.data.X_std
            w_alpha = self.nullsp["w_alpha_std"]
            w_beta = self.nullsp["w_beta_std"]
        else:
            X = self.data.X_
            w_alpha = self.nullsp["w_alpha"]
            w_beta = self.nullsp["w_beta"]
        fig, ax = plot_nullspace_correction(
            w_alpha,
            w_beta,
            self.nullsp["v_"],
            self.nullsp["gamma"],
            X,
            self.data.x,
            self.data.y_,
            name=title,
            coef_name_alpha=self.nullsp["w_alpha_name"],
            coef_name_beta=self.nullsp["w_beta_name"],
            max_gamma=self.max_gamma,
            con_val=self.con_val,
            method=self.opt_gamma_method,
        )
        return fig, ax

    def scatter_predictions(self, std=False, title="", ax=None, return_fig=False):
        """Method that scatters based on nullspace correction."""
        y_ = self.data.y_

        if std:
            X = self.data.X_std - self.data.X_std.mean(axis=0)
            w_alpha = self.nullsp["w_alpha_std"]
            w_beta = self.nullsp["w_beta_std"]
        else:
            X = self.data.X_ - self.data.X_.mean(axis=0)
            w_alpha = self.nullsp["w_alpha"]
            w_beta = self.nullsp["w_beta"]

        w = [(w_alpha + self.nullsp["v_"][-1, :]), w_alpha, w_beta]
        labels = [
            r"$\mathbf{X}(\beta + \mathbf{v})$",
            r"$\mathbf{X}\beta$",
            r"$\mathbf{X}\beta_{T1}$",
        ]

        return scatter_predictions(X, y_, w, labels, ax=ax, return_fig=return_fig)


# Todo: Theres something wrong with this function. Should be dubugged if used or deleted.
if 0:

    def nullspace_correction_wrap(w_alpha, w_beta, dml_obj, verbose=True, std=False):
        """Function that calls 'nullspace_correction allowing to shorten syntax and use SynMLData class.

        Parameters
        ----------
        w_alpha : ndarray
            Regression coefficients obtained with method alpha
        w_beta : ndarray
            Regression coefficients obtained with method beta
        dml_obj : obj
            Object from SynMLData class
        std : bool, default=False
            Indicate whether Standardization(Z-Scoring) shall be performed.
            If yes, use X_std of the dml_object, if not use X.

        Returns
        -------
        see 'nullspace_correction'
        """
        if std:
            X = dml_obj.X_std
        else:
            X = dml_obj.X_
        x = dml_obj.x

        y_ = dml_obj.y_
        y_range = np.max(y_) - np.min(y_)
        min_exp = -5
        max_exp = np.floor(np.log10(int((10**2) * y_range)))
        gs = np.logspace(min_exp, max_exp, 30)
        gs = np.append(gs, [int((10**2) * y_range)])
        return Nullspace.nullspace_correction(
            w_alpha, w_beta, X, x, gs=gs, comp_block=0, verbose=verbose
        )
