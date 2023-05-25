import numpy as np

import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def optimize_pls(
    X,
    y,
    max_comps=20,
    folds=10,
    nb_stds=1,
    min_distance_search=False,
    featlin=0,
    verbose=False,
    **kwargs,
):
    """Optimize the number of components for PLS regression."""

    components = np.arange(1, max_comps + 1).astype("uint8")
    rmse = np.zeros((len(components),))
    stds = np.zeros((len(components),))
    dist_l2 = []
    # Loop through all possibilities
    for comp in components:
        pls = PLSRegression(n_components=comp, scale=False)

        # Cross-validation: Predict the test samples based on a predictor that was trained with the
        # remaining data. Repeat until prediction of each sample is obtained.
        # (Only one prediction per sample is allowed)
        # Only these two cv methods work. Reson: Each sample can only belong to EXACTLY one test set.
        # Other methods of cross validation might violate this constraint
        # For more information see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
        scores = cross_val_score(
            pls, X, y, cv=folds, n_jobs=-1, scoring="neg_root_mean_squared_error"
        )
        rmse[comp - 1] = -scores.mean()
        stds[comp - 1] = scores.std()

        if min_distance_search:
            # Find the PLS vector that has minimal L2 distance to the featlin vector.
            # Comparing these two vector can subsequently tell us, whether we're close and the feature should be considered or not.
            pls = PLSRegression(n_components=comp, scale=False)
            reg = pls.fit(X, y)
            diff_vec = featlin - reg.coef_.reshape(-1)
            dist_l2.append(np.linalg.norm(diff_vec, ord=2))

    if min_distance_search:
        dist_l2 = np.array(dist_l2)
        l2_min_loc = np.argmin(dist_l2)
        l2_dist_min_comp = components[l2_min_loc]

    rmsemin_loc = np.argmin(rmse)
    rmsemin_param = components[rmsemin_loc]

    # Extract the components that are within the standard deviation of the minimum
    filtered_lst = [
        (i, element)
        for i, element in enumerate(rmse)
        if element < rmse[rmsemin_loc] + (nb_stds * stds[rmsemin_loc])
    ]
    rmse_std_min_loc, rmse_std_min = min(filtered_lst)
    rmse_std_min_param = components[rmse_std_min_loc]

    # Train model with optimal number of components
    pls = PLSRegression(n_components=rmsemin_param, scale=False)
    reg = pls.fit(X, y)
    # Extract the coefficients
    coef_cv = reg.coef_

    # Train model with std min number of components
    pls = PLSRegression(n_components=rmse_std_min_param, scale=False)
    reg = pls.fit(X, y)
    # Extract the coefficients
    coef_std_cv = reg.coef_

    cv_res_dict = {
        "rmse_vals": rmse,
        "rmse_std": stds,
        "components": components,
        "rmse_std_min": rmse_std_min,
        "rmse_std_min_param": rmse_std_min_param,
        "rmse_min_param": rmsemin_param,
        "ceof_std_cv": coef_std_cv,
        "coef_cv": coef_cv,
    }

    # Train the model with min distance number of components
    if min_distance_search:
        pls = PLSRegression(n_components=l2_dist_min_comp, scale=False)
        reg = pls.fit(X, y)
        # Extract the coefficients
        coef_min_dist = reg.coef_

    if min_distance_search:
        dist_l2_res_dict = {
            "l2_distance": dist_l2,
            "l2_min_param": l2_dist_min_comp,
            "l2_min_loc": l2_min_loc,
            "components": components,
            "coef_min_dist": coef_min_dist,
        }
        return {
            "cv_res": cv_res_dict,
            "l2_distance_res": dist_l2_res_dict,
            "algorithm": "PLS",
        }
    return {"cv_res": cv_res_dict, "algorithm": "PLS"}


def optimize_rr(
    X,
    y,
    alpha_lim: list = None,
    folds=5,
    nb_stds=1,
    plot=False,
    min_distance_search=True,
    featlin: list = None,
    verbose=False,
):
    """Crossvalidation of RR algorithm and plotting of results"""

    if alpha_lim is None:
        alpha_lim = [10e-5, 10e3]
    if featlin is None:
        featlin = []

    nb_iterations = 20
    nb_selected_values = 8
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
        # Define the cross validation
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        # Define the model
        ridge = Ridge()
        # Define the grid search
        grid = GridSearchCV(
            estimator=ridge,
            param_grid=dict(alpha=alpha),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        # Fit the grid search
        grid.fit(X, y)
        # Obtain all the results
        results = grid.cv_results_
        rmse.append(np.sqrt(-results["mean_test_score"]))
        stds.append(results["std_test_score"])

        # Obtain the two alpha values with the lowest mean test score
        idx = np.argpartition(-results["mean_test_score"], 2)

        alpha_lim_cv = [results["param_alpha"][idx[0]], results["param_alpha"][idx[1]]]
        # Sort the two alpha values
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

    # Train the model with the optimal alpha value
    ridge = Ridge(alpha=alpha_opt_cv)
    ridge.fit(X, y)
    # Obtain the coefficients
    coef_cv = ridge.coef_
    # Train the model with the optimal std alpha value
    ridge = Ridge(alpha=alpha_std_opt_cv)
    ridge.fit(X, y)
    # Obtain the coefficients
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

    # Rerun the entire loops if the min distance search is required
    # Unfortunately, this is not very efficient, but necessary fro now to obtain the min distance
    # TODO: find a way to optimize this
    if min_distance_search:
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
                # y_hat = ridge.predict(X)
                diff_vec = featlin - ridge.coef_.reshape(-1)
                dist_l2_ = np.linalg.norm(diff_vec, ord=2)
                # if min_dist_alpha is None:
                #    min_dist_alpha = a
                #    min_dist = dist_l2_
                # else:
                #    if dist_l2_ < min_dist:
                #        min_dist_alpha = a
                #        min_dist = dist_l2_
                dist_l2_iteration_i.append(dist_l2_)
            # Define the new grid
            # Sort the dist_l2s of the last iteration
            dist_l2.append(dist_l2_iteration_i)
            sorted_norms = np.sort(dist_l2_iteration_i)
            try:
                alpha_min = alphas_[
                    np.where(dist_l2_iteration_i == sorted_norms[0])[0][0] - 1
                ]
            except:  # noqa TODO: fix this
                alpha_min = alphas_[
                    np.where(dist_l2_iteration_i == sorted_norms[0])[0][0]
                ]
            # Making it more robust to look into a space that is a bit larger than just between the wo best values.
            try:
                alpha_min2 = alphas_[
                    np.where(dist_l2_iteration_i == sorted_norms[0])[0][0] + 1
                ]
            except:  # noqa
                alpha_min2 = alphas_[
                    np.where(dist_l2_iteration_i == sorted_norms[1])[0][0]
                ]

            alphas_ = np.geomspace(alpha_min, alpha_min2, num=nb_selected_values)

        all_alphas = np.concatenate(alphas_l2, axis=0)
        all_dist = np.concatenate(dist_l2, axis=0)
        l2_min_loc = np.argmin(all_dist)
        alpha_min_l2 = all_alphas[l2_min_loc]
        # Train the model with the min dist alpha value
        ridge = Ridge(alpha=alpha_min_l2)
        ridge.fit(X, y)
        # Obtain the coefficients
        coef_min_dist = ridge.coef_

        dist_l2_res_dict = {
            "alphas": all_alphas,
            "l2_distance": dist_l2,
            "l2_min_param": alpha_min_l2,
            "l2_min_loc": l2_min_loc,
            "coef_min_dist": coef_min_dist,
        }
        return {
            "cv_res": cv_res_dict,
            "l2_distance_res": dist_l2_res_dict,
            "algorithm": "RR",
        }

    return {"cv_res": cv_res_dict, "algorithm": "RR"}


def optimise_pls_cv(
    X,
    y,
    max_comps=20,
    folds=10,
    plot_components=False,
    std=False,
    min_distance_search=False,
    featlin=[],
):
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

    Returns
    -------
    rmse : ndarray
        mean of rmse for all folds for each number of comp
    components : ndarray
        list of components tested for cv
    """

    components = np.arange(1, max_comps + 1).astype("uint8")
    rmse = np.zeros((len(components),))
    stds = np.zeros((len(components),))
    l2_distance = np.zeros((len(components),))
    if std:
        X = StandardScaler().fit_transform(X)

    # Loop through all possibilities
    for comp in components:
        pls = PLSRegression(n_components=comp, scale=False)

        # Cross-validation: Predict the test samples based on a predictor that was trained with the
        # remaining data. Repeat until prediction of each sample is obtained.
        # (Only one prediction per sample is allowed)
        # Only these two cv methods work. Reson: Each sample can only belong to EXACTLY one test set.
        # Other methods of cross validation might violate this constraint
        # For more information see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
        scores = cross_val_score(
            pls, X, y, cv=folds, n_jobs=-1, scoring="neg_mean_squared_error"
        )
        rmse[comp - 1] = -scores.mean()
        stds[comp - 1] = scores.std()
        if min_distance_search:
            # Find the PLS vector that has minimal L2 distance to the featlin vector.
            # Comparing these two vector can subsequently tell us, whether we're close and the feature should be considered or not.
            reg = pls.fit(X - np.mean(X, axis=0), y - y.mean())
            diff_vec = featlin - reg.coef_.reshape(-1)
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
        with plt.style.context(("ggplot")):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(components, rmse, "-o", color="blue", mfc="blue", label="Mean RMSE")
            ax.plot(components, rmse - stds, color="k", label="Mean RMSE - 1 std")
            ax.plot(components, rmse + stds, color="k", label="Mean RMSE + 1 std")
            ax.plot(
                components[rmsemin_loc],
                rmse[rmsemin_loc],
                "P",
                ms=10,
                mfc="red",
                label="Lowest RMSE",
            )
            ax.plot(
                components[rmse_std_min],
                rmse[rmse_std_min],
                "P",
                ms=10,
                mfc="green",
                label=f"Within {nb_stds} std of best numebr of comp.",
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
            ax.set_xlabel("Number of PLS components")
            ax.set_ylabel("RMSE")
            ax.set_title("PLS Crossvalidation")
            ax.set_xlim(left=0.5)
            ax.legend()

    res_dict = {
        "rmse_vals": rmse,
        "components": components,
        "rmse_std_min": rmse_std_min,
        "l2_distance": np.array(l2_distance),
    }
    return res_dict


def nrmse(
    y: np.array = None, y_pred: np.array = None, X: np.array = None, beta: np.array = None
):
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
    if y_pred is None:
        y_pred = X @ beta

    return 100 * mean_squared_error(y, y_pred, squared=False) / (np.max(y) - np.min(y))
