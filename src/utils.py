# Author: Joachim Schaeffer, 2023, joachim.schaeffer@posteo.de
import numpy as np

import matplotlib.pyplot as plt  # type: ignore
from sklearn.cross_decomposition import PLSRegression  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.model_selection import cross_val_score, cross_validate  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore


def optimize_rr_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_lim: list = [10e-5, 10e3],
    folds: int = 5,
    nb_iterations: int = 20,
    nb_selected_values: int = 8,
    nb_stds: int = 1,
    verbose: bool = False,
) -> dict:
    """Crossvalidation of RR algorithm and plotting of results"""
    rmse = []
    stds = []
    alphas = []

    alpha_lim_cv = alpha_lim
    # Refine iteratively, by cutting the search space in half
    for i in range(nb_iterations):
        # Define the search space by selecting 4 alpha values, equally spaced in log space
        if i == 0:
            alpha = np.logspace(
                np.log10(alpha_lim_cv[0]), np.log10(alpha_lim_cv[1]), nb_selected_values
            )
        else:
            alpha = np.logspace(
                np.log10(alpha_lim_cv[0]),
                np.log10(alpha_lim_cv[1]),
                nb_selected_values - 2,
            )
        alphas.append(alpha)

        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        ridge = Ridge()
        grid = GridSearchCV(
            estimator=ridge,
            param_grid=dict(alpha=alpha),
            cv=cv,
            scoring="neg_mean_squared_error",
        )

        grid.fit(X, y)
        results = grid.cv_results_
        rmse.append(np.sqrt(-results["mean_test_score"]))
        stds.append(results["std_test_score"])

        # Obtain the two alpha values with the lowest mean test score
        idx = np.argpartition(-results["mean_test_score"], 2)

        alpha_lim_cv = [results["param_alpha"][idx[0]], results["param_alpha"][idx[1]]]
        alpha_lim_cv.sort()

        # If the two alpha values are close enough, stop the search
        if verbose:
            print(f"Iteration: {i} done")

        # Break if the relative difference between the two rmse values associated with the two alpha values is small enough
        if (
            np.abs(rmse[i][idx[0]] - rmse[i][idx[1]])
            / np.max([rmse[i][idx[0]], rmse[i][idx[1]]])
            < 0.001
        ):
            if verbose:
                print(f"Converged after {i} iterations, breaking")
            break
    alphas_cv = np.concatenate(alphas, axis=0)
    rmse_cv = np.concatenate(rmse, axis=0)
    stds_cv = np.concatenate(stds, axis=0)
    rmsemin_loc_cv = np.argmin(rmse_cv)
    id_min_cv = np.argmin(rmse_cv)
    alpha_opt_cv = alphas_cv[id_min_cv]
    nb_stds = 1

    filtered_lst = [
        (i, element)
        for i, element in zip(alphas_cv, rmse_cv)
        if element < rmse_cv[rmsemin_loc_cv] + (nb_stds * stds_cv[rmsemin_loc_cv])
    ]
    _, rmse_std_min = max(filtered_lst)
    # Return the alpha value corresponding to the std rule from the filtered list
    # index of the alpha value corresponding to the std rule
    idx = [i for i, element in enumerate(rmse_cv) if element == rmse_std_min][0]
    alpha_std_opt_cv = alphas_cv[idx]

    ridge = Ridge(alpha=alpha_opt_cv)
    ridge.fit(X, y)
    coef_cv = ridge.coef_

    ridge = Ridge(alpha=alpha_std_opt_cv)
    ridge.fit(X, y)
    coef_std_cv = ridge.coef_

    cv_res_dict = {
        "rmse_vals": rmse_cv,
        "rmse_std": stds_cv,
        "alphas": alphas_cv,
        "rmse_std_min": rmse_std_min,
        "rmse_std_min_param": alpha_std_opt_cv,
        "rmse_min_param": alpha_opt_cv,
        "ceof_std_cv": coef_std_cv,
        "coef_cv": coef_cv,
    }
    return cv_res_dict


def optimize_rr_min_dist(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_lim: list = None,
    nb_iterations: int = 20,
    nb_selected_values: int = 8,
    beta_prop: np.ndarray = None,
) -> dict:
    """Unfortunately, this is not very efficient, but works for now to obtain the min distance"""

    alphas_l2 = []
    dist_l2 = []
    # min_dist_alpha = None
    # min_dist = None
    # Define the search space by selecting 4 alpha values, equally spaced in log space
    alphas_ = np.logspace(
        np.log10(alpha_lim[0]), np.log10(alpha_lim[1]), nb_selected_values
    )

    for i in range(nb_iterations):
        dist_l2_iteration_i = []
        alphas_l2.append(alphas_)
        # Define the model
        ridge = Ridge()
        # Minimum distance search
        for j, a in enumerate(alphas_):
            ridge = Ridge(alpha=a)
            ridge.fit(X, y)
            diff_vec = beta_prop - ridge.coef_.reshape(-1)
            dist_l2_ = np.linalg.norm(diff_vec, ord=2)
            dist_l2_iteration_i.append(dist_l2_)
        # Define the new grid
        # Sort the dist_l2s of the last iteration
        dist_l2.append(dist_l2_iteration_i)
        sorted_norms = np.sort(dist_l2_iteration_i)
        try:
            alpha_min = alphas_[
                np.where(dist_l2_iteration_i == sorted_norms[0])[0][0] - 1
            ]
        except:  # noqa Fix this
            alpha_min = alphas_[np.where(dist_l2_iteration_i == sorted_norms[0])[0][0]]
        # Making it more robust to look into a space that is a bit larger than just between the wo best values.
        try:
            alpha_min2 = alphas_[
                np.where(dist_l2_iteration_i == sorted_norms[0])[0][0] + 1
            ]
        except:  # noqa Fix this
            alpha_min2 = alphas_[np.where(dist_l2_iteration_i == sorted_norms[1])[0][0]]

        alphas_ = np.geomspace(alpha_min, alpha_min2, num=nb_selected_values)

    all_alphas = np.concatenate(alphas_l2, axis=0)
    all_dist = np.concatenate(dist_l2, axis=0)
    l2_min_loc = np.argmin(all_dist)
    alpha_min_l2 = all_alphas[l2_min_loc]

    ridge = Ridge(alpha=alpha_min_l2)
    ridge.fit(X, y)
    coef_min_dist = ridge.coef_

    dist_l2_res_dict = {
        "alphas": all_alphas,
        "l2_distance": dist_l2,
        "l2_min_param": alpha_min_l2,
        "l2_min_loc": l2_min_loc,
        "coef_min_dist": coef_min_dist,
    }
    return dist_l2_res_dict


def neg_mse_exp_y_scorer(model, X, y):
    y_pred = model.predict(X)
    return -(mean_squared_error(np.exp(y), np.exp(y_pred)))


def optimize_pls_cv(
    X: np.ndarray,
    y: np.ndarray,
    max_comps: int = 20,
    folds: int = 10,
    plot_components: bool = False,
    std: bool = False,
    min_distance_search: bool = False,
    beta_prop: np.ndarray = None,
    neg_rmse_exp_scorer: bool = False
) -> dict:
    """Crossvalidation of PLS algorithm and plotting of results.

    Parameters
    ----------
    X : ndarray
        2D array of training data
    y : ndarray
        1D array of responses
    max_comps : int, default=20
        maximum number of PLS components for cv
    folds : int, default=10
        number of folds for crossvalidation
    plot_components : bool, default=False
        Indicate whether to plot results
    std : bool, default=False
        Inidcates whether to standardize/z-score X
    """

    components = np.arange(1, max_comps + 1).astype("uint8")
    rmse = np.zeros((len(components),))
    stds = np.zeros((len(components),))
    l2_distance = np.zeros((len(components),))
    # if std:
    #    X = StandardScaler().fit_transform(X)

    # Loop through all possibilities
    for comp in components:
        pls = PLSRegression(n_components=comp, scale=std)

        if neg_rmse_exp_scorer:
            cv_results = cross_validate(
                estimator=pls,
                X=X,
                y=y,
                groups=None,
                scoring=neg_mse_exp_y_scorer,
                cv=folds,
                n_jobs=-1,
                verbose=0,
            )
        else:
            cv_results = cross_validate(
                estimator=pls,
                X=X,
                y=y,
                groups=None,
                scoring="neg_mean_squared_error",
                cv=folds,
                n_jobs=-1,
                verbose=0,
            )

        rmse[comp - 1] = -cv_results["test_score"].mean()
        stds[comp - 1] = cv_results["test_score"].std()

        if min_distance_search:
            # Find the PLS vector that has minimal L2 distance to the beta_prop vector.
            # Comparing these two vector can subsequently tell us, whether we're close and the feature should be considered or not.
            reg = pls.fit(X - np.mean(X, axis=0), y - y.mean())
            diff_vec = beta_prop - reg.coef_.reshape(-1)
            l2_distance[comp - 1] = np.linalg.norm(diff_vec, 1)

    if min_distance_search:
        l2_min_loc = np.argmin(l2_distance)

    rmsemin_loc = np.argmin(rmse)
    # Minimum number of componets where rms is still < rmse[rmsemin_loc]+stds[rmsemin_loc]
    nb_stds = 1

    filtered_lst = [
        (i, element)
        for i, element in enumerate(rmse)
        if element < rmse[rmsemin_loc] + (nb_stds * stds[rmsemin_loc])
    ]
    rmse_std_min, _ = min(filtered_lst)
    if plot_components is True:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(components, rmse, "-o", color="blue", mfc="blue", label="Mean RMSE")
        ax.plot(components, rmse - stds, color="k", label="Mean RMSE +- 1 std")
        ax.plot(components, rmse + stds, color="k", label="")
        ax.plot(
            components[rmsemin_loc],
            rmse[rmsemin_loc],
            "P",
            ms=10,
            mfc="red",
            label="Min. RMSE",
            markeredgecolor="red",
        )
        # marker outline color transparent
        ax.plot(
            components[rmse_std_min],
            rmse[rmse_std_min],
            "P",
            ms=10,
            mfc="green",
            label=f"{nb_stds} standard-error rule",
            markeredgecolor="green",
        )
        if min_distance_search:
            ax.plot(
                components[l2_min_loc],
                rmse[l2_min_loc],
                "P",
                ms=10,
                mfc="black",
                label="Smallest L1 distance to passed feature",
            )
        ax.set_xticks(components)
        ax.set_xlabel("PLS components")
        ax.set_ylabel("RMSE")
        if std:
            title = f"{folds}-Fold CV, PLS, Z-Scored X"
        else:
            title = f"{folds}-Fold CV, PLS"
        ax.set_title(title)
        ax.set_xlim(left=0.5)
        ax.legend()
        fig_name = title.replace(' ', '')
        fig_name = fig_name.replace(',', '')
        plt.savefig(f"results/Nullspace/{fig_name}.pdf")
        plt.show()

    res_dict = {
        "rmse_vals": rmse,
        "components": components,
        "rmse_std_min": rmse_std_min,
        "l2_distance": np.array(l2_distance),
    }
    return res_dict


def nrmse(
    y: np.ndarray,
    y_pred: np.ndarray = None,
    X: np.ndarray = None,
    beta: np.ndarray = None,
) -> float:
    """Normalized root mean squared error

    Parameters
    ----------
    y : array-like
        Target values
    X : array-like
        Data matrix
    beta : array-like
        Regression coefficients
    """
    if y_pred is None and (X is not None and beta is not None):
        y_pred = X @ beta

    return 100 * mean_squared_error(y, y_pred, squared=False) / (np.max(y) - np.min(y))
