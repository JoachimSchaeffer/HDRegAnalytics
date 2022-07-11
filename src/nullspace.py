import numpy as np
from numpy import linalg as LA

from scipy import linalg
from scipy.linalg import toeplitz

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib import cm   
import matplotlib.cm as cmx
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

from src.helper import truncate_colormap

def nullspace_correction(w_alpha, w_beta, X, Vdlin, gs=[None], comp_block=0, snig=0):
    """This functions performs a nulspace normalization of regression coefficents.

    Inputs:
    w_alpha: linearize feature coefficient vector 
    w_beta: regression coefficient vector from (latent) variable method 
    X: Data matrix that was used for estimating the regression coefficeint
    Predictions can be made via np.dot(X, w_alpha) or X@w_alpha
    p: vector of penalizations of deviations form the nulspace vector
    
    Return: 
    w_nulspace: vector that is, depending on p, close to the nulspace. 
    The problem is set up such that the L2 norm differences between ther regression 
    coefficients is minimzed. If vecotrs close to the nulspace are used, they are penalized by p. 
    
    Purpose: 
    
    """
    points = len(w_alpha)
    if gs[0]==None:
        gs = np.logspace(-5, 3, 30)
        gs = np.append(gs, [2500, 5000, np.int((10**4)*(np.max(X)-np.min(X)))])
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

        plt.plot(Vdlin, v_, label='Simplified Equations')
        plt.plot(Vdlin, v, label='Non Simplified Equations')
        plt.legend()
        plt.show()

    # Approach of Toeplitz matrix regularization 

    # r1 = np.concatenate((np.array([1]), np.zeros(points-2)))
    # c1 = np.concatenate((np.array([1, -1]), np.zeros(points-2)))
    # E1 = toeplitz(r1, c1)

    r2 = np.concatenate((np.array([-1, 0, 0]), np.zeros(points-5)))
    c2 = np.concatenate((np.array([-1, 2, -1]), np.zeros(points-3)))
    E2 = toeplitz(r2, c2)
    
    nb_gs = len(gs)
    v_ = np.zeros((nb_gs, shape[1]))
    v_snig = np.zeros((nb_gs, shape[1]))
    norm_ = np.zeros(nb_gs)
    norm_snig= np.zeros(nb_gs)
    g2 = 1000
    
    for i,g in enumerate(gs):
        v_[i,:] = -linalg.inv(g*X.T@X+I)@w
        norm_[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
        if snig:
            v_snig[i,:] = -linalg.inv(g*X.T@X+g2*E2.T@E2+I)@w 
            norm_snig[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
    if snig:   
        return v, v_, v_snig, norm_, norm_snig, gs
    else:
        return v, v_, norm_, gs


def plot_nullspace_correction(w_alpha, w_beta, v, gs, X, x, name='', coef_name_alpha='', coef_name_beta='', return_fig=1):
    """Plot the nullspace correction


    """
    plt.style.use('./styles/plots-latex.mplstyle')
    points = len(w_alpha)
    color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
    cNorm  = mcolors.Normalize(vmin=0, vmax=np.log(gs.max()))
    
    cmap = truncate_colormap(cm.get_cmap('plasma'), 0.1, 0.9)
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
  
    for i in range(v.shape[0]):
        ax[1].plot(x, w_alpha+v[i,:], color=scalarMap.to_rgba(np.log(gs[i])))

    ax[1].plot(x, w_alpha, label=coef_name_alpha, color='g')   
    ax[1].plot(x, w_beta, label=coef_name_beta , color='k')
    
    ax[1].fill_between(x.reshape(-1), w_alpha, y2=w_alpha+v[-1,:], hatch='oo', zorder=2, fc=(1, 1, 1, 0.8), label=r'Appr. contained in $N(X)$')
    
    ax[1].set_xlabel('x values')
    ax[1].set_ylabel(r'Regression Coefficients $(\bm\beta)$')

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
 
    cb.set_label(r'$\ln(\gamma)$', labelpad=10)

    ax[0].grid()    
    ax[1].grid()
    ax[1].legend(loc=2)
    fig.suptitle(name)

    plt.show()
    if return_fig:
        return fig, ax
    else:
        return None