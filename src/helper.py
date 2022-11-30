import numpy as np
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import src.basis as basis
import src.nullspace


# Helper functions for this notebook
def construct_data(x_min, x_max, basis_function,  
                        mean_params, stdv_params,
                        num_datapoints=50, draws=10, plot_results=False): 
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
    obj = basis.BasicsData(basis_function, num_basis).Phi(x)

    # Draw the parameters for the matrix
    # m = np.random.uniform(low=range_m[0], high=range_m[1], size=rows)
    param_vals = np.zeros((draws, len(mean_params)))
    for i,(j,k) in enumerate(zip(mean_params, stdv_params)):
        param_vals[:, i] = np.array([np.random.normal(loc=j, scale=k) for p in range(draws)])

    # Construct it
    obj = obj.construct_X_data(param_vals)
    if plot_results:
        # Plot it# Construct it
        plt.plot(x, obj.X.T)
        plt.title('Data Generated from Basis Function')
        plt.show()

    return obj

def construct_plot_data_interactive(
        x_min, x_max, basis_function,  
        mean_param0, mean_param1, mean_param2, 
        stdv_params0, stdv_params1, stdv_params2, 
        num_datapoints=50, draws=10):
    """Wraper around 'construct_plot_data' to interact with ipython widget
    """
    mean_params = np.array([mean_param0, mean_param1, mean_param2])     
    stdv_params = np.array([stdv_params0, stdv_params1, stdv_params2])   

    basis_obj = construct_data(x_min, x_max, basis_function,  
                        mean_params, stdv_params,
                        num_datapoints=num_datapoints, draws=draws, plot_results=1)
    return None  

def plot_x_tt2(X, x, ax, color, labelx, labely, label_data='Train', zorder=1, **kwargs): 
    """Plot Data"""
    # Get linestyle kwarg if it exists
    if 'linestyle' in kwargs:
        linestyle = kwargs['linestyle']
    else:
        linestyle = '-'
    ax.plot(x, X[:, :].T, linestyle, label=label_data, lw=1, color=color, zorder=zorder)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=4)
    #axs.set_title('Training Data')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    return ax

def plot_corrheatmap(ax, x, X, cmap, label, title, cols=True): 
    if cols:
        X_df = pd.DataFrame(X[:, ::10])
        x = x[::10]
    else: 
        X_df = pd.DataFrame(X[:, :])
    X_corr = np.abs(X_df.corr())
    if cols:
        X_corr.set_index(np.round(x, 1), inplace=True)
        X_corr.set_axis(np.round(x, 1), axis='columns', inplace=True)
    mask = np.triu(X_corr)
    if cols: 
        ax = sns.heatmap(
            X_corr, 
            vmin=0, vmax=1, center=0.4,
            cmap=cmap,
            square=True, 
            xticklabels=100,
            yticklabels=100,
            ax = ax,
            mask=mask
        )
    else:
        ax = sns.heatmap(
            X_corr, 
            vmin=0.82, vmax=1, center=0.91,
            cmap=cmap,
            square=True,
            xticklabels=10,
            yticklabels=10,
            ax = ax,
            mask=mask
        )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    ax.set_yticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel(label)
    # axs[0, 1].set_xticks(np.range(0, len(X_corr)), labels=range(2011, 2019))
    return ax

def nullspace_correction_wrap(w_alpha, w_beta, dml_obj, std=False):
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
    max_exp = np.floor(np.log10(int((10**2)*y_range)))
    gs  = np.logspace(min_exp, max_exp, 30)
    gs = np.append(gs, [int((10**2)*y_range)])
    return src.nullspace.nullspace_correction(w_alpha, w_beta, X, x, gs=gs, comp_block=0)


def optimize_pls(X, y, max_comps=20, folds=10, nb_stds=1, min_distance_search=False, 
        featlin=0, **kwargs):
    """Optimize the number of components for PLS regression."""

    components = np.arange(1, max_comps + 1).astype('uint8')
    rmse = np.zeros((len(components), ))
    stds = np.zeros((len(components), ))
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
        scores = cross_val_score(pls, X, y, cv=folds, n_jobs=-1, scoring='neg_root_mean_squared_error')
        rmse[comp - 1] = -scores.mean()
        stds[comp - 1] = scores.std()

        if min_distance_search: 
            # Find the PLS vector that has minimal L2 distance to the featlin vector.
            # Comparing these two vector can subsequently tell us, whether we're close and the feature should be considered or not.
            pls = PLSRegression(n_components=comp, scale=False)
            reg = pls.fit(X, y)  
            diff_vec = featlin-reg.coef_.reshape(-1)
            dist_l2.append(np.linalg.norm(diff_vec, ord=2))

    if min_distance_search: 
        dist_l2 = np.array(dist_l2)
        l2_min_loc = np.argmin(dist_l2)
        l2_dist_min_comp = components[l2_min_loc]
    
    rmsemin_loc = np.argmin(rmse)
    rmsemin_param = components[rmsemin_loc]
    
    # Extract the components that are within the standard deviation of the minimum
    filtered_lst = [(i, element) for i,element in enumerate(rmse) if element < rmse[rmsemin_loc]+(nb_stds*stds[rmsemin_loc])]
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

    cv_res_dict = {'rmse_vals': rmse, 'rmse_std': stds, 'components': components, 
        'rmse_std_min': rmse_std_min, 'rmse_std_min_param': rmse_std_min_param, 'rmse_min_param': rmsemin_param,
        'ceof_std_cv': coef_std_cv, 'coef_cv': coef_cv}

    # Train the model with min distance number of components
    if min_distance_search:
        pls = PLSRegression(n_components=l2_dist_min_comp, scale=False)
        reg = pls.fit(X, y)
        # Extract the coefficients
        coef_min_dist = reg.coef_

    if min_distance_search: 
        dist_l2_res_dict = {'l2_distance': dist_l2, 'l2_min_param': l2_dist_min_comp, 
            'l2_min_loc': l2_min_loc, 'components': components, 'coef_min_dist': coef_min_dist}
        return {'cv_res': cv_res_dict, 'l2_distance_res': dist_l2_res_dict, 'algorithm': 'PLS'}
    return {'cv_res': cv_res_dict, 'algorithm': 'PLS'}


def optimize_rr(X, y, alpha_lim: list=None, folds=5, nb_stds=1, plot=False, min_distance_search=True, std=False, featlin: list=None):
    """Crossvalidation of RR algorithm and plotting of results"""

    if alpha_lim is None:
        alpha_lim = [10e-5, 10e3]
    if featlin is None:
        featlin = []

    nb_iterations = 15
    nb_selected_values = 8
    rmse = []
    stds = []
    alphas = []

    if std: 
        X = StandardScaler().fit_transform(X)
    alpha_lim_cv = alpha_lim
    # Refine iteratively, by cutting the search space in half
    for i in range(nb_iterations): 
        # Define the search space by selecting 4 alpha values, equally spaced in log space
        if i == 0:
            alpha = np.logspace(np.log10(alpha_lim_cv[0]), np.log10(alpha_lim_cv[1]), nb_selected_values)
        else:
            alpha = np.logspace(np.log10(alpha_lim_cv[0]), np.log10(alpha_lim_cv[1]), nb_selected_values-2)
        alphas.append(alpha)
        # Define the cross validation
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        # Define the model
        ridge = Ridge()
        # Define the grid search
        grid = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alpha), cv=cv, scoring='neg_mean_squared_error')
        # Fit the grid search
        grid.fit(X, y)
        # Obtain all the results
        results = grid.cv_results_
        rmse.append(np.sqrt(-results['mean_test_score']))
        stds.append(results['std_test_score'])

        # Obtain the two alpha values with the lowest mean test score
        idx = np.argpartition(-results['mean_test_score'], 2)

        alpha_lim_cv = [results['param_alpha'][idx[0]], results['param_alpha'][idx[1]]]
        # Sort the two alpha values
        alpha_lim_cv.sort()

        # If the two alpha values are close enough, stop the search
        print(f'Iteration: {i} done')

        # Break if the relative difference between the two rmse values associated with the two alpha values is small enough
        if np.abs(rmse[i][idx[0]]-rmse[i][idx[1]])/np.max([rmse[i][idx[0]], rmse[i][idx[1]]]) < 0.001:
            print(f'Converged after {i} iterations')
            break
    alphas_cv = np.concatenate(alphas, axis=0)
    rmse_cv = np.concatenate(rmse, axis=0)
    stds_cv = np.concatenate(stds, axis=0)
    rmsemin_loc_cv = np.argmin(rmse_cv)
    id_min_cv = np.argmin(rmse_cv)
    alpha_opt_cv = alphas_cv[id_min_cv]
    nb_stds = 1 

    filtered_lst = [(i, element) for i,element in zip(alphas_cv, rmse_cv) if element < rmse_cv[rmsemin_loc_cv]+(nb_stds*stds_cv[rmsemin_loc_cv])]
    _, rmse_std_min = max(filtered_lst)
    # Return the alpha value corresponding to the std rule from the filtered list
    # index of the alpha value corresponding to the std rule
    idx = [i for i,element in enumerate(rmse_cv) if element == rmse_std_min][0]
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

    cv_res_dict = {'rmse_vals': rmse_cv, 'rmse_std': stds_cv, 'alphas':alphas_cv, 
        'rmse_std_min': rmse_std_min, 'rmse_std_min_param': alpha_std_opt_cv, 'rmse_min_param': alpha_opt_cv,
        'ceof_std_cv': coef_std_cv, 'coef_cv': coef_cv}

    # Rerun the entire loops if the min distance search is required
    # Unfortunately, this is not very efficient, but necessary fro now to obtain the min distance
    # TODO: find a way to optimize this
    if min_distance_search: 
        alphas_l2 = []
        dist_l2 = []
        # Define the search space by selecting 4 alpha values, equally spaced in log space
        alphas_ = np.logspace(np.log10(alpha_lim[0]), np.log10(alpha_lim[1]), nb_selected_values)

        for i in range(nb_iterations): 
            alphas_l2.append(alphas_)
            # Define the cross validation
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
            # Define the model
            ridge = Ridge()
            # Minimum distance search
            for j, a in enumerate(alphas_):
                ridge = Ridge(alpha=a)
                ridge.fit(X, y)
                # y_hat = ridge.predict(X)
                diff_vec = featlin-ridge.coef_.reshape(-1)
                dist_l2_= np.linalg.norm(diff_vec, ord=2)
                try:
                    if dist_l2_ <= np.min(dist_l2):
                        min_dist_alpha = a
                except:
                    min_dist_alpha = -999
                dist_l2.append(dist_l2_)
            # Define the new grid
            # Sort the dist_l2s of the last iteration
            sorted_norms = np.sort(dist_l2[-j:])
            try:
                alpha_min = alphas_[np.where(dist_l2[-j:]==sorted_norms[0])[0][0]-2]
            except:
                alpha_min = alphas_[np.where(dist_l2[-j:]==sorted_norms[0])[0][0]]
            # Making it more robust to look into a space that is a bit larger than just between the wo best values. 
            try:
                alpha_min3 = alphas_[np.where(dist_l2[-j:]==sorted_norms[1])[0][0]+2]
            except:
                alpha_min3 = alphas_[np.where(dist_l2[-j:]==sorted_norms[3])[0][0]]
            alphas_ = np.geomspace(alpha_min, alpha_min3, num=nb_selected_values)

        l2_alphas = np.concatenate(alphas_l2, axis=0)
        dist_l2 = np.array(dist_l2)
        l2_min_loc = np.argmin(dist_l2)
        # Train the model with the min dist alpha value
        ridge = Ridge(alpha=min_dist_alpha)
        ridge.fit(X, y)
        # Obtain the coefficients
        coef_min_dist = ridge.coef_
    
        dist_l2_res_dict = {'alphas': l2_alphas, 'l2_distance': dist_l2, 'l2_min_param': min_dist_alpha, 
            'l2_min_loc': l2_min_loc, 'coef_min_dist': coef_min_dist}
        return {'cv_res': cv_res_dict, 'l2_distance_res': dist_l2_res_dict, 'algorithm': 'RR'}

    return {'cv_res': cv_res_dict, 'algorithm': 'RR'}

def optimize_cv(
        X, y, max_comps=20, alpha_lim: list=None, folds=10, nb_stds=1, 
        plot_components=False, std=False, min_distance_search=False, 
        featlin=0, algorithm='PLS', **kwargs):
    """Crossvalidation or optimization of regression coefficient distance for PLS or RR
    
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
    nb_stds : int, default=1
        Choose highest regularization, where rms is still < rmse[rmsemin_loc]+nb_stds*stds[rmsemin_loc]
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
    if alpha_lim is None:
        alpha_lim = [10e-5, 10e3]

    if std: 
        X = StandardScaler().fit_transform(X)

    if algorithm=='PLS':
        res_dict = optimize_pls(X, y, max_comps=max_comps, folds=folds, nb_stds=nb_stds, 
        plot_components=plot_components, std=std, min_distance_search=min_distance_search, 
        featlin=featlin, **kwargs)
 
    elif algorithm=='RR':
        res_dict = optimize_rr(X, y, alpha_lim=alpha_lim, folds=folds, nb_stds=nb_stds, 
        std=std, min_distance_search=min_distance_search, 
        featlin=featlin, **kwargs)

    # If kwarg plot is TRUE, plot the results
    if kwargs.get('plot', False):
        key = 'components' if algorithm=='PLS' else 'alphas'
        plot_cv_results(res_dict, key=key)
    return res_dict


def plot_cv_results(res_dict, key='components'):
    """Plot the results of the cross validation function"""
    colors_IBM = ['#648fff', '#785ef0', '#dc267f', '#fe6100', '#ffb000',  '#000000']
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(res_dict['cv_res'][key], res_dict['cv_res']['rmse_vals'], color = 'blue', label='RMSE')
    ax[0].scatter(res_dict['cv_res'][key], res_dict['cv_res']['rmse_vals']+res_dict['cv_res']['rmse_std'], color = 'k', label='RMSE + STD')
    ax[0].scatter(res_dict['cv_res'][key], res_dict['cv_res']['rmse_vals']-res_dict['cv_res']['rmse_std'], color = 'k')
    # Scatter a circle around the minimum
    ax[0].scatter(res_dict['cv_res']['rmse_min_param'], np.min(res_dict['cv_res']['rmse_vals']), color=colors_IBM[1], s=100)
    # Scatter a circle around the mean rmse that is still within 1 std of the minimum
    ax[0].scatter(res_dict['cv_res']['rmse_std_min_param'], res_dict['cv_res']['rmse_std_min'], color=colors_IBM[2], s=100)
    

    ax[0].set_xlabel(f'Number of {key}')
    ax[0].set_ylabel('RMSE')
    ax[0].set_title(f'RMSE vs. Number of {key}')
    ax[0].legend()
    
    ax[1].scatter(res_dict['l2_distance_res'][key], res_dict['l2_distance_res']['l2_distance'], color=colors_IBM[0], label='L2 Distance')
    # Scatter a circle around the minimum
    min_l2_alpha = res_dict['l2_distance_res']['l2_min_param']
    min_l2_dist = np.min(res_dict['l2_distance_res']['l2_distance'])
    ax[1].scatter(min_l2_alpha, min_l2_dist, 
        color=colors_IBM[1], s=100, label=f'Min. L2 Dist. {min_l2_dist:.2f} {key} {min_l2_alpha:.2f}')
    ax[1].set_xlabel(f'Number of {key}')
    ax[1].set_ylabel('L2 Distance')
    ax[1].set_title(f'L2 Distance vs. Number of {key}')
    ax[1].legend()
    
    # Set axes log scale if the key is alpha
    if key == 'alphas':
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_xlabel(f'RR alpha')
        ax[1].set_xlabel(f'RR alpha')
    plt.tight_layout()
    plt.show()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncates a colormap. This is important because a) many people are partly colorblind and a lot of 
    colormaps unsuited for them, and b) a lot of colormaps include yellow whichcanbe hard to see on some 
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
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Legacy
# Not used anymore
def optimize_regcoef_dist(model, X, y, regularization_limits, lin_coef_, norm=1, max_depth=5): 
    '''Find learned regression coefficients that lead to prediciotns as close as possible to the desired NRMSE error.
    As of now, only PLS regression or ridge regression are implemented. 
    This algorithm starts with the highest regularization and goes down stepwise. In case of PLS in steps of componets, 
    in the case of ridge regression in 10 gemetrically spaced values between the regularization limits.
    '''
    if model=='PLS': 
        # Start with highest regularization and decrease by step of 1.
        if type(regularization_limits[0]) != int:
            raise TypeError('If PLS is the model, the upper regularization error must be integer!')
        norms = np.full((regularization_limits[0]), np.inf)
        for i in range(regularization_limits[0]):
            model = PLSRegression(n_components=i+1, tol=1e-7, scale=False)
            model.fit(X-np.mean(X, axis=0), y-y.mean())
            norm_i = np.linalg.norm(lin_coef_-model.coef_.reshape(-1), norm)
            if norm_i <= np.min(norms):
                reg = i+1
            norms[i] = norm_i
    elif model=='ridge':
        # While PLS is very easy, due to the low amount of regularization parameters, things get more complicated with PLS
        # We will be stepwise refining the grid, by taking the two smallest distnaces and then go down until the list is exhausted.
        alphas = np.geomspace(regularization_limits[0], regularization_limits[1], num=11)
        for i in range(max_depth):
            norms = np.full(len(alphas), np.inf)
            for i, alpha in enumerate(alphas):
                model = Ridge(alpha=alpha)
                model.fit(X-np.mean(X, axis=0), y-y.mean())
                norm_i = np.linalg.norm(lin_coef_-(model.coef_.reshape(-1)), norm)
                if norm_i <= np.min(norms):
                    reg = alpha
                norms[i] = norm_i
            # plt.plot(alphas, norms)
            # plt.show()
            sorted_norms = np.sort(norms)
            try:
                alpha_min = alphas[np.where(norms==sorted_norms[0])[0][0]-2]
            except:
                alpha_min = alphas[np.where(norms==sorted_norms[0])[0][0]]
            # Making it more robust to look into a space that is a bit larger than just between the wo best values. 
            try:
                alpha_min3 = alphas[np.where(norms==sorted_norms[1])[0][0]+2]
            except:
                alpha_min3 = alphas[np.where(norms==sorted_norms[3])[0][0]]
            alphas = np.geomspace(alpha_min, alpha_min3, num=15)
 
    else: 
        raise ValueError('Not Implemented')
    return reg