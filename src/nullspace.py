import numpy as np
from numpy import linalg as LA

from scipy import linalg
from scipy.linalg import toeplitz

# from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib import cm   
import matplotlib.cm as cmx
from matplotlib import rc
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

import src.helper

       
class Nullspace():
    """Synthethic Machine Learning Data class. this class inherits from basis (takes Phi and X from basis)
    This class can generate response variables (y).
    """
    def __init__(self, data, **kwargs):
        self.std = None
        self.weights = {}                 # Dictionary to store weights learned from mean subtracted data X_
        self.nullsp = {}
        self.data = data                  # Should be a BasicsCalss object or duck type
        self.max_nrmse = None
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
                self.weights[string_id[i] + ' std'] = coef_std
                self.weights[string_id[i] + ' std retrans'] = coef_std/self.data.stdx
            except:
                self.weights[string_id[i] + ' std'] = 'Undefined Std = 0 for some column'
                self.weights[string_id[i] + ' std retrans'] = 'Undefined Std = 0 ofr some column'
        return self
    
    def nullspace_correction(
        self, w_alpha=None, w_alpha_name=None, w_beta=None, w_beta_name=None, std=False, max_nrmse=-0.5,
        plot_results=False, save_plot=False, path_save='', file_name='', **kwargs):
        """Function that calls 'nullspace_correction allowing to shorten syntax and use SynMLData class.

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

        if (w_alpha is None) & ('key_alpha' not in kwargs):
            NameError('Either w_alpha is passed directly or key for learned coefficients must be passed')

        if (w_beta is None) & ('key_beta' not in kwargs):
            NameError('Either w_alpha is passed directly or key for learned coefficients must be passed')
    
        self.nullsp['w_beta_name'] = w_beta_name
        self.nullsp['w_alpha_name'] = w_alpha_name

        # To keep things simple, standardized weights also included here! 
        # In case std==False this is not efficient, because the standardized coef. aren't used. 
        # But it's easier to understand and less verbose. 
        if w_alpha is None:
            self.nullsp['w_alpha'] = self.weights[kwargs.get('key_alpha')]
            self.nullsp['w_alpha_std'] = self.weights[kwargs.get('key_alpha') + ' std']
        else:
            self.nullsp['w_alpha'] = w_alpha
            self.nullsp['w_alpha_std'] = w_alpha * self.data.stdx

        if w_beta is None:
            self.nullsp['w_beta'] = self.weights[kwargs.get('key_beta')]
            self.nullsp['w_beta_std'] = self.weights[kwargs.get('key_beta') + ' std']
        else:
            self.nullsp['w_beta'] = w_beta
            self.nullsp['w_beta_std'] = w_beta * self.data.stdx

        if std: 
            X = self.data.X_std
            self.nullsp['w_alpha_name'] += ' std'
            self.nullsp['w_beta_name'] += ' std'
            key_alpha = 'w_alpha_std'
            key_beta = 'w_beta_std'
        else:
            X = self.data.X_
            key_alpha = 'w_alpha'
            key_beta = 'w_beta'
            
        x = self.data.x
        self.nullsp['info'] = ''

        if 'nb_gammas' in kwargs:
            nb_gammas = kwargs.get('nb_gammas')
        else:
            nb_gammas = 30

        y_ = self.data.y_

        # Simple approach to set gamma
        # y_range = np.max(y_) - np.min(y_)
        # min_exp = -5
        # max_exp = np.floor(np.log10(int((10**2)*y_range)))

        # gamma_vals = np.geomspace(10**(-5), (10**2)*y_range, nb_gammas)
        # gamma_vals  = np.logspace(min_exp, max_exp, nb_gammas)
        # gamma_vals = np.append(gamma_vals, [int((10**2)*y_range)])
        
        nrmse_alpha = 100*mean_squared_error(y_, X@(self.nullsp[key_alpha]), squared=False)/(np.max(y_)-np.min(y_))
        nrmse_beta = 100*mean_squared_error(y_, X@(self.nullsp[key_beta]), squared=False)/(np.max(y_)-np.min(y_))
        if max_nrmse < 0:
            self.max_nrmse = np.abs(max_nrmse) * np.abs(nrmse_alpha-nrmse_beta)
        else: 
            self.max_nrmse = max_nrmse
        # print(np.max(y_)-np.min(y_))
        #print(self.max_nrmse)
        #print(nrmse_alpha)
        #print(nrmse_beta)
        #print(np.abs(nrmse_alpha-nrmse_beta))
        if np.abs(nrmse_alpha-nrmse_beta) > self.max_nrmse:
            # Find largest value for gamma that is not modifying the resutls more than max_nrmse. 
            # The issue is that this problem is not necessarily convex and the optimizer get stuck in local optima.
            # This is a simple (but ineffective way) way of optimizing which should suffice for this issue.
            nb_gammas=10
            # gammas = np.append([0], np.geomspace(1, 10**10, nb_gammas))
            # gammas = np.linspace(10**8, 1, nb_gammas)
            # nrmse_diff = np.zeros(len(gammas))
            # depth=5
            # for i in range(depth):
                 # print(i)
            #    for j, gamma in enumerate(gammas): 
            #        nrmse_diff[j] = find_gamma(
            #            [gamma], self.nullsp[key_alpha], self.nullsp[key_beta], X, x, y_, self.max_nrmse)
            #        if np.abs(nrmse_diff[j]) < self.max_nrmse: 
            #            break
            #    if j==0:
            #        break
            #    else:
            #        if gammas[j-1]==0:
            #            # gammas = np.append([0], np.geomspace(10**(-12), gammas[j], nb_gammas))
            #            gammas = np.linspace(0, gammas[j], nb_gammas)
            #       else:
            #            gammas=np.linspace(gammas[j-1], gammas[j], nb_gammas)
                        # gammas=np.geomspace(gammas[j-1], gammas[j], nb_gammas)
            #print(i)
            #self.max_gamma = gamma

            # y_range = np.max(y_) - np.min(y_)
            # gs_inital = 100*y_range
            # Find value for gamma that 
            import scipy as sp
            gamma_upper_limit = sp.optimize.minimize(
                find_gamma_, 100, args=(self.nullsp[key_alpha], self.nullsp[key_beta], X, x, y_, max_nrmse),
                method='Nelder-Mead', bounds=[(1, 10**10)], options={'xatol' : 0.01})
            self.max_gamma = gamma_upper_limit.x[0]

            print(f'Gamma value corresponding to nrmse={self.max_nrmse:.2f} % is {self.max_gamma:.3f}')

            if self.max_gamma < 10**(-12):
                gamma_vals = [self.max_gamma]
            else: 
                gamma_vals = np.geomspace(10**(-12), self.max_gamma+2*(10**(-12)), 30)

            self.nullsp['v'], self.nullsp['v_'], self.nullsp['norm_'], self.nullsp['gamma'] = nullspace_correction(
                self.nullsp[key_alpha], self.nullsp[key_beta], X, x, gs=gamma_vals, comp_block=0)
        else:
            self.nullsp['v'] = np.array([self.nullsp[key_beta]-self.nullsp[key_alpha]])
            self.nullsp['v_'] = np.array(self.nullsp[key_beta]-[self.nullsp[key_alpha]])
            self.nullsp['gamma'] = 0 
            self.max_gamma = np.inf
        print(self.max_gamma)
        if plot_results:
            fig, ax = self.plot_nullspace_correction(std=std)
            if save_plot:
                fig.savefig(path_save + file_name)
            return self, fig, ax 
        else:
            return self

    def plot_nullspace_correction(self, std=False, title=''):
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
            w_alpha = self.nullsp['w_alpha_std']
            w_beta = self.nullsp['w_beta_std']
        else: 
            X = self.data.X_
            w_alpha = self.nullsp['w_alpha']
            w_beta = self.nullsp['w_beta']
        fig, ax = plot_nullspace_correction(
                w_alpha, w_beta, self.nullsp['v_'], self.nullsp['gamma'],
                X, self.data.x, self.data.y_, name=title, 
                coef_name_alpha=self.nullsp['w_alpha_name'], coef_name_beta=self.nullsp['w_beta_name'], 
                max_nrmse=self.max_nrmse, max_gamma=self.max_gamma)
        return fig, ax
    

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

def find_gamma(gs, w_alpha, w_beta, X_, x, y_, nrmse): 
    v, v_, norm_, gs = nullspace_correction(w_alpha, w_beta, X_, x, gs=gs, comp_block=False)
    nrmse_reg = 100*mean_squared_error(y_, X_@(w_alpha), squared=False)/(np.max(y_)-np.min(y_))
    nrmse_nulls = 100*mean_squared_error(y_, X_@(w_alpha+v_.reshape(-1)), squared=False)/(np.max(y_)-np.min(y_))
    print(np.abs(nrmse_reg-nrmse_nulls))
    return np.abs(nrmse_reg-nrmse_nulls)

def find_gamma_(gs, w_alpha, w_beta, X_, x, y_, nrmse): 
    v, v_, norm_, gs = nullspace_correction(w_alpha, w_beta, X_, x, gs=gs, comp_block=False)
    nrmse_reg = 100*mean_squared_error(y_, X_@(w_alpha), squared=False)/(np.max(y_)-np.min(y_))
    nrmse_nulls = 100*mean_squared_error(y_, X_@(w_alpha+v_.reshape(-1)), squared=False)/(np.max(y_)-np.min(y_))
    print(np.abs(nrmse_reg-nrmse_nulls))
    return ((nrmse_reg-nrmse_nulls)**2 - nrmse**2)**2

def format_label(max_gamma, max_nrmse):
    """Helps with label formatting for the nullspace!
    """
    if max_gamma < 0.01:
        g_str = f'{max_gamma:.3f}'
    else: 
        g_str = f'{max_gamma:.2f}'
    
    # The gamma value will highly depend on the magnitude of the differenc eof the features/regression coefficients
    # Thus it is very difficult to interpret.
    if max_nrmse <= 0.01:
        if max_nrmse <= 10**(-8):
            label=r'$\in \mathcal{\mathbf{N}}(X)$ enlarged by 0.00% nrmse'
        else:
            label=r'$\in \mathcal{\mathbf{N}}(X)$' + f' enlarged by {max_nrmse:.3f}% nrmse'# + '\n Corresponding ' + r'$\gamma=$' + g_str
    else:
        label=r'$\in \mathcal{\mathbf{N}}(X)$' + f' enlarged by {max_nrmse:.2f}% nrmse'# + '\n Corresponding ' + r'$\gamma=$' + g_str

    return label

def plot_nullspace_correction(
    w_alpha, w_beta, v, gs, X, x, y, name='', coef_name_alpha='', coef_name_beta='', return_fig=True, 
    max_nrmse=-9999, max_gamma=-9999):
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
    max_nrmse : float, default=-9999
        Maximum nrmse diff. that was allowed.
    gamma : float, default=-9999
        Gamma value correponding to maximum nrmse


    Returns
    -------
    fig : object
        matplotlib figure object
    ax : object 
    """
    y = y-np.mean(y)
    X = X - np.mean(X, axis=0)
    color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
    nrmse_vals = [
        100*mean_squared_error(y, X@(w_alpha+v[-1,:]), squared=False)/(np.max(y)-np.min(y)),
        100*mean_squared_error(y, X@(w_alpha+v[0,:]), squared=False)/(np.max(y)-np.min(y))]
    nrmse_min = np.min(nrmse_vals)
    nrmse_max = np.max(nrmse_vals)
    cNorm  = mcolors.Normalize(vmin=nrmse_min, vmax=nrmse_max)

    # Outdated
    # For synthethic data nrmse_min approx equal nrmse_max and numeric precision might lead to nrmse_min > nrmse_max
    # eps = 10**(-12)
    # if np.abs(nrmse_min-nrmse_max) > eps:
    #     cNorm  = mcolors.Normalize(vmin=nrmse_min, vmax=nrmse_max)
    # else:
    #     cNorm  = mcolors.Normalize(vmin=nrmse_min-eps, vmax=nrmse_max+eps)
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

    # Initializing the nrmse error to 2, makes it easy to spot issues
    nrmse = 2*np.ones(v.shape[0])
    for i in range(v.shape[0]):
        nrmse[i] = mean_squared_error(y, X@(w_alpha+v[i,:]), squared=False)/(np.max(y)-np.min(y))  
        ax[1].plot(x, w_alpha+v[i,:], color=scalarMap.to_rgba(100*nrmse[i]), zorder=i)

    markevery = int(len(x)/15)
    nrmse_alpha = 100*mean_squared_error(y, X@(w_alpha), squared=False)/(np.max(y)-np.min(y))
    nrmse_beta = 100*mean_squared_error(y, X@(w_beta), squared=False)/(np.max(y)-np.min(y))

    coef_alpha_label = f"{coef_name_alpha}, nrmse: {nrmse_alpha:.2f} %"
    coef_beta_label = f"{coef_name_beta}, nrmse: {nrmse_beta:.2f} %"
    ax[1].plot(x, w_alpha, label=coef_alpha_label, color='darkgreen', marker="P", markevery=markevery, markersize=8, linewidth=2.5, zorder=v.shape[0]+1)   
    ax[1].plot(x, w_beta, label=coef_beta_label, color='k', linewidth=2.5, zorder=v.shape[0]+1)

    # ax[1].fill_between(x.reshape(-1), w_alpha, y2=w_alpha+v[-1,:], hatch='oo', zorder=-1, fc=(1, 1, 1, 0.8), label=r'Appr. contained in $N(X)$')
    if max_gamma < 0.01:
        g_str = f'{max_gamma:.3f}'
    else: 
        g_str = f'{max_gamma:.2f}'
    
    label = format_label(max_gamma, max_nrmse)

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
    # cb.set_label(r'NRMSE($\mathbf{X}\boldsymbol{\beta}_{a+v(\gamma)}, \mathbf{X}\boldsymbol{\beta}_a$)', labelpad=10)
    cb.set_label(r'NRMSE (%)', labelpad=10)

    ax[0].grid()    
    ax[1].grid()
    ax[1].legend(loc=2)
    fig.suptitle(name)
    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None