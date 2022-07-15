import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cross_decomposition import PLSRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import src.basis as basis
import src.nullspace


# Helper functions for this notebook
def construct_data(x_min, x_max, basis_function,  
                        mean_params, stdv_params, objtype='data',
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
    objtype: str, default='data'
        'data' will initialize from the simple data class, 'mldata' from the mldataclass including an objective y.
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
    if objtype=='data':
        obj = basis.Basis(basis_function, num_basis).Phi(x)
    elif objtype=='mldata':
        obj = basis.SynMLData(basis_function, num_basis).Phi(x)
    else:
        ValueError(f'You entered {objtype}, which does not correspond to an implemented option.')

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


def construct_plot_data_interactive(x_min, x_max, basis_function,  
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
    return src.nullspace.nullspace_correction(w_alpha, w_beta, X, x, gs=gs, comp_block=0, snig=0)


def optimise_pls_cv(X, y, max_comps=20, folds=10, plot_components=False, std=False):
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
    
    components = np.arange(1, max_comps + 1).astype('uint8')
    rmse = np.zeros((len(components), ))
    stds = np.zeros((len(components), ))
    
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
        scores = cross_val_score(pls, X, y, cv=folds, n_jobs=-1, scoring='neg_mean_squared_error')
        rmse[comp - 1] = -scores.mean()
        stds[comp - 1] = scores.std()

    rmsemin_loc = np.argmin(rmse)
    # Minimum number of componets where rms is still < rmse[rmsemin_loc]+stds[rmsemin_loc]
    
    filtered_lst = [(i, element) for i,element in enumerate(rmse) if element < rmse[rmsemin_loc]+stds[rmsemin_loc]]
    rmse_std_min, _ = min(filtered_lst)
    if plot_components is True:
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots()
            ax.plot(components, rmse, '-o', color = 'blue', mfc='blue', label='Mean RMSE')
            ax.plot(components, rmse-stds, color = 'k', label='Mean RMSE - 1 std')
            ax.plot(components, rmse+stds, color = 'k', label='Mean RMSE + 1 std')
            ax.plot(components[rmsemin_loc], rmse[rmsemin_loc], 'P', ms=10, mfc='red', label='Lowest RMSE')
            ax.plot(components[rmse_std_min], rmse[rmse_std_min], 'P', ms=10, mfc='green', label='Within 1 std of best numebr of comp.')
            
            ax.set_xticks(components)
            ax.set_xlabel('Number of PLS components')
            ax.set_ylabel('RMSE')
            ax.set_title('PLS Crossvalidation')
            ax.set_xlim(left=0.5)
            ax.legend()
    
    return rmse, components


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