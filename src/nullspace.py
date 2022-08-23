import numpy as np
from numpy import linalg as LA

from scipy import linalg
from scipy.linalg import toeplitz

# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib import cm   
import matplotlib.cm as cmx
from matplotlib import rc
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

import src.helper

def nullspace_correction(w_alpha, w_beta, X, x, gs=[None], comp_block=False):
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
    comp_block : bool, default=False
        Use component block method

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
    points = len(w_alpha)
    if gs is None:
        gs = np.geomspace(10**(-5), (10**4)*(np.max(X)-np.min(X)), 30)
        # gs = np.logspace(-5, 3, 30)
        # gs = np.append(gs, [2500, 5000, np.int((10**4)*(np.max(X)-np.min(X)))])
    # difference between coefficients
    w = w_alpha - w_beta
    # SVD decomposition of X
    U, s_, Vh = linalg.svd(X)
    
    # Build helper vectors and matrices, not optimized for speed or memory usage!
    shape = X.shape
    ssti = np.zeros((shape[0], shape[0]))
    s = np.zeros((shape[0], shape[1]))
    sst = np.zeros((shape[0], shape[0]))
    st = np.zeros((shape[1], shape[0]))
    I = np.identity(shape[1])

    np.fill_diagonal(sst, (s_*s_))
    ssti = np.linalg.inv(sst)
    np.fill_diagonal(s, s_)
    np.fill_diagonal(st, s_)

    # Do the magic:
    
    # Approach 1: Full equations
    S = st@ssti@s
    v = (Vh.T@S@Vh - I)@w
    
    # Approch 2: working with block matrices. 
    # This doesnt solve the issue of poor conditioned X....
    if comp_block:
        v11 = Vh.T[:41,:41]
        v12 = Vh.T[:41,41:]
        v21 = Vh.T[41:,:41]
        v22 = Vh.T[41:,41:]
        left = np.concatenate((-v12@v12.T, -v22@v12.T), axis=0)
        right = np.concatenate((-v12@v22.T, -v22@v22.T), axis=0)
        v_ = np.concatenate((left, right), axis=1)@w

        plt.plot(x, v_, label='Simplified Equations')
        plt.plot(x, v, label='Non Simplified Equations')
        plt.legend()
        plt.show()

    # Approach of Toeplitz matrix regularization 
    # r1 = np.concatenate((np.array([1]), np.zeros(points-2)))
    # c1 = np.concatenate((np.array([1, -1]), np.zeros(points-2)))
    # E1 = toeplitz(r1, c1)

    # r2 = np.concatenate((np.array([-1, 0, 0]), np.zeros(points-5)))
    # c2 = np.concatenate((np.array([-1, 2, -1]), np.zeros(points-3)))
    # E2 = toeplitz(r2, c2)
    
    nb_gs = len(gs)
    v_ = np.zeros((nb_gs, shape[1]))
    norm_ = np.zeros(nb_gs)
    
    # v_snig = np.zeros((nb_gs, shape[1]))
    # norm_snig= np.zeros(nb_gs)
    # g2 = 1000
    
    for i,g in enumerate(gs):
        v_[i,:] = -linalg.inv(g*X.T@X+I)@w
        norm_[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
        # if snig:
        #    v_snig[i,:] = -linalg.inv(g*X.T@X+g2*E2.T@E2+I)@w 
        #    norm_snig[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
    #if snig:   
    #    return v, v_, v_snig, norm_, norm_snig, gs
    #else:
    
    return v, v_, norm_, gs

def find_gamma(gs, w_alpha, w_beta, X_, x, y_, mape): 
    v, v_, norm_, gs = nullspace_correction(w_alpha, w_beta, X_, x, gs=gs, comp_block=False)
    mape_reg = 100*mean_absolute_percentage_error(y_, X_@(w_alpha))  
    mape_nulls = 100*mean_absolute_percentage_error(y_, X_@(w_alpha+v_.reshape(-1)))
    return np.abs(mape_reg-mape_nulls)

def plot_nullspace_correction(
    w_alpha, w_beta, v, gs, X, x, y, name='', coef_name_alpha='', coef_name_beta='', return_fig=True, 
    max_mape=-9999, max_gamma=-9999):
    """Plot the nullspace correction

    Parameters 
    ----------
    w_alpha : ndarray
        1D array of linear feature coefficient vector 
    w_beta : ndarray
        1D array of regression coefficients
    v : ndarray
        1D array of of regression coefficeint contianed in teh nullspace that minimize the L2-norm
    gs : ndarray
        1D array of penalizations of deviations form the nulspace vector    
        Either equal to inpiuts or derived in the function    
    X : ndarray 
        2D Data matrix that was used for estimating the regression coefficeint
        Predictions can be made via np.dot(X, w_alpha) or X@w_alpha
    x : ndarray
        1D array of values of the smooth domain
    name : str, default=''
        Name of the Plot, suptitle of the Matplotlib function
    coef_name_alpha : str, defualt='' 
        Description/label for coefficients alpha
    coef_name_beta : str, defailt=''
        Description/label for coefficients beta
    return_fig : bool, default=True
        Indicatees whetehr function returns figure or not
    max_mape : float, default=-9999
        Maximum MAPE diff. that was allowed.
    gamma : float, default=-9999
        Gamma value correponding to maximum MAPE


    Returns
    -------
    fig : object
        matplotlib figure object
    ax : object 

    """
    color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
    mape_vals = [
        100*mean_absolute_percentage_error(y, X@(w_alpha+v[-1,:])),
        100*mean_absolute_percentage_error(y, X@(w_alpha+v[0,:]))]
    mape_min = np.min(mape_vals)
    mape_max = np.max(mape_vals)
    cNorm  = mcolors.Normalize(vmin=mape_min, vmax=mape_max)

    # Outdated
    # For synthethic data mape_min approx equal mape_max and numeric precision might lead to mape_min > mape_max
    # eps = 10**(-12)
    # if np.abs(mape_min-mape_max) > eps:
    #     cNorm  = mcolors.Normalize(vmin=mape_min, vmax=mape_max)
    # else:
    #     cNorm  = mcolors.Normalize(vmin=mape_min-eps, vmax=mape_max+eps)
    # cNorm  = mcolors.Normalize(vmin=0, vmax=np.log(gs.max()))
    
    cmap = src.helper.truncate_colormap(cm.get_cmap('plasma'), 0.1, 0.7)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    figsize = [11, 13]
    fig, ax = plt.subplots(2,1, figsize=figsize, constrained_layout=True, sharex=True)
    
    ax[0].plot(x, X[:, :].T, label='Train', lw=1, color=color_list[0])
    ax[0].set_title('Data')
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), loc=2)
    ax[0].set_ylabel('y values')

    y_min = ax[0].get_ylim()[0]
    y_max = ax[0].get_ylim()[1]
    ax[0].vlines(0, y_min, y_max, colors='k', linestyles='solid', linewidths=0.8)
    ax[0].hlines(0, min(x), max(x), colors='k', linestyles='solid', linewidths=0.8)
    ax[0].set_ylim(y_min, y_max)

    # Initializing the mape error to 2, makes it easy to spot issues
    mape = 2*np.ones(v.shape[0])
    for i in range(v.shape[0]):
        mape[i] = mean_absolute_percentage_error(y, X@(w_alpha+v[i,:]))  
        ax[1].plot(x, w_alpha+v[i,:], color=scalarMap.to_rgba(100*mape[i]), zorder=i)

    markevery = int(len(x)/15)
    mape_alpha = 100*mean_absolute_percentage_error(y, X@(w_alpha))
    mape_beta = 100*mean_absolute_percentage_error(y, X@(w_beta))

    coef_alpha_label = f"{coef_name_alpha}, MAPE: {mape_alpha:.2f} %"
    coef_beta_label = f"{coef_name_beta}, MAPE: {mape_beta:.2f} %"
    ax[1].plot(x, w_alpha, label=coef_alpha_label, color='darkgreen', marker="P", markevery=markevery, markersize=8, linewidth=2.5, zorder=v.shape[0]+1)   
    ax[1].plot(x, w_beta, label=coef_beta_label, color='k', linewidth=2.5, zorder=v.shape[0]+1)

    # ax[1].fill_between(x.reshape(-1), w_alpha, y2=w_alpha+v[-1,:], hatch='oo', zorder=-1, fc=(1, 1, 1, 0.8), label=r'Appr. contained in $N(X)$')
    if max_mape <= 0.01:
        if max_mape <= 10**(-8):
            label=r'$\in \mathcal{\mathbf{N}}(X)$ enlarged by 0.00% MAPE'
        else:
            label=r'$\in \mathcal{\mathbf{N}}(X)$' + f' enlarged by {max_mape:.1e}% MAPE' + '\n Corresponding ' + r'$\gamma=$' + f'{max_gamma:.2f}'
    else:
        label=r'$\in \mathcal{\mathbf{N}}(X)$' + f' enlarged by {max_mape:.2f}% MAPE' + '\n Corresponding ' + r'$\gamma=$' + f'{max_gamma:.2f}'

    ax[1].fill_between(
        x.reshape(-1), w_alpha, y2=w_alpha+v[-1,:], color='darkgrey', zorder=-1, alpha=0.8, label=label)
            
    
    ax[1].set_xlabel('x values')
    ax[1].set_ylabel(r'Regression Coefficients $(\beta)$')

    # Set bottom and left spines as x and y axes of coordinate system
    y_min = ax[1].get_ylim()[0]
    y_max = ax[1].get_ylim()[1]
    ax[1].vlines(0, y_min, y_max, colors='k', linestyles='solid', linewidths=0.8)
    ax[1].hlines(0, min(x), max(x), colors='k', linestyles='solid', linewidths=0.8)

    ax[0].set_xlim(min(x), max(x))
    ax[1].set_ylim(y_min, y_max)
    ax[1].set_title('Nullspace Perspective')

    cb = fig.colorbar(cm.ScalarMappable(norm=cNorm, cmap=cmap), 
                        ax=ax[1], pad=0.01)

    # cb.set_label(r'$\ln(\gamma)$', labelpad=10)
    # cb.set_label(r'MAPE($\mathbf{X}\boldsymbol{\beta}_{a+v(\gamma)}, \mathbf{X}\boldsymbol{\beta}_a$)', labelpad=10)
    cb.set_label(r'MAPE (%)', labelpad=10)

    ax[0].grid()    
    ax[1].grid()
    ax[1].legend(loc=2)
    fig.suptitle(name)
    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None