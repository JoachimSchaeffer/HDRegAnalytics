# This source file is a collection of helper functions 
# for the nullspace normalization.
# some of them could potentially be intergrated in the oobject oriented structure
# for experimental reasons this was not done yet
import numpy as np
from numpy import linalg as LA

from scipy import linalg

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib import cm   # ToDo: clean up!
import matplotlib.cm as cmx
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
plt.style.use('./styles/plots-latex.mplstyle')

import src.src_lin_feature
from src.src_lin_feature import truncate_colormap

def nul_space_basis_correction(w_alpha, w_beta, X, Vdlin, gs=[0.05, 0.1, 1], comp_block=0):
    '''
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
    This functions performs a nulspace normalization of regression coefficents. 
    '''
    points = len(w_alpha)
    gs = np.linspace(0.1, 100, 500)
    gs = np.append(gs, [250, 500, 1000,2500, 5000, 10000])
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

    # Do the magic, more comments needed here!
    
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
    from scipy.linalg import toeplitz

    r1 = np.concatenate((np.array([1]), np.zeros(points-2)))
    c1 = np.concatenate((np.array([1, -1]), np.zeros(points-2)))

    r2 = np.concatenate((np.array([-1, 0, 0]), np.zeros(points-5)))
    c2 = np.concatenate((np.array([-1, 2, -1]), np.zeros(points-3)))

    E1 = toeplitz(r1, c1)
    E2 = toeplitz(r2, c2)
    
    nb_gs = len(gs)
    v_ = np.zeros((nb_gs, shape[1]))
    v_snig = np.zeros((nb_gs, shape[1]))
    norm_ = np.zeros(nb_gs)
    norm_snig= np.zeros(nb_gs)
    g2 = 1000
    
    for i,g in enumerate(gs):
        v_[i,:] = -linalg.inv(g*X.T@X+I)@w
        v_snig[i,:] = -linalg.inv(g*X.T@X+g2*E2.T@E2+I)@w 
        norm_[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
        norm_snig[i] = LA.norm(w_alpha+v_[i,:]-w_beta, 2)
        
    
    return v, v_, v_snig, norm_, norm_snig, gs


def plot_nulspace_approximations(w_alpha, w_beta, v, norm, gs, name='', model='', path='', save=0):
    
    points = len(w_alpha)
    Vdlin = np.linspace(3.5,2,points)
    cNorm  = mcolors.Normalize(vmin=0, vmax=np.log(gs.max()))
    
    cmap = truncate_colormap(cm.get_cmap('plasma'), 0.1, 0.9)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    fig, ax = plt.subplots() 
    for i in range(v.shape[0]):
        ax.plot(Vdlin, w_alpha+v[i,:], color=scalarMap.to_rgba(np.log(gs[i])))

    ax.plot(Vdlin, w_alpha, label=model, color='g')   
    ax.plot(Vdlin, w_beta, label=r'$\bm\beta_{Tlin}$ Log Variance', color='k')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Regression Coefficient')

    cb = fig.colorbar(cm.ScalarMappable(norm=cNorm, cmap=cmap), 
                          ax=ax, pad=0.01)
 
    if name[:4] in ['plss', 'rrsn']: 
        cb.set_label(r'$\ln(\gamma_1)$', labelpad=10)
    else:
        cb.set_label(r'$\ln(\gamma)$', labelpad=10)
        
    plt.grid()
    plt.legend(loc=2)
    if save: 
        plt.savefig(path + name + '.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return None