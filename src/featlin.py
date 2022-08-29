"""
Featue Linearization Methodology
Fuctions to linearize nonlinear features and 
Subsequently finding a constant term via regeression to match the metdoch
"""

# Packages
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from src.basis import BasicsData
from src.nullspace import Nullspace

from src.helper import optimise_pls_cv
from src.helper import optimize_regcoef_mape
from src.helper import optimize_regcoef_dist

import jax.numpy as jnp
from jax import grad
from jax import jacfwd


class Featlin(): 
    """Class that performs feature linearization, comparison and potentially selection.
    By choice we decided not to inherit other classes in here, but instead realy on composition.
    """
    def __init__(self, X=None, x=None, y=None, data_obj=None, feat_funcs=None):
        """Initiate Featlin object. 
        Either pass X, x, y, where 'x' is either the domain of your measurements (e.g. 2.0V-3.5V or Hz fro spectra)
        You can also pass a linearly spaced vecotr x, if the functional domain is less well defined. 
        Only the plots will be affected by x.

        Parameters
        ----------
        feat_funcs : list of jax numpy functions
            feature functions that shall be testes and among which the algorithm will decide.
        
        """

        if X is None or y is None:
            if data_obj is None: 
                raise ValueError('You must pass either X and y, or data_obj to instatiate class object')
            else:
                self.data = data_obj
        else:
            if x is None:
                x = np.linspace(0, X.shape[1]-1, X.shape[1])
            self.data = BasicsData(X=X, x=x, y=y)

        if feat_funcs is None: 
            feat_fun = [
                lambda a : jnp.mean(a),
                lambda a : jnp.sum(a**2),
                lambda a : jnp.var(a),
                lambda x: jax_moment(x,3)/((jax_moment(x,2))**(3/2)),
                lambda x: jax_moment(x,4)/(jax_moment(x,2)**2)-3
                ]
            feat_fun_names = [
                'Sum',
                'Sum of Squares',
                'Variance', 
                'Skewness',
                'Kurtosis',
                ]
            self.feat_fun = feat_fun
            self.feat_fun_names = feat_fun_names
        else:
            self.feat_fun = feat_funcs['functions']
            self.feat_fun = feat_funcs['names']
        
        self.feat_fun_np = [lambda a: np.array(self.feat_fun[i](a)) for i in range(len(self.feat_fun))]

        self.metric = None              # Metric that shall be used for 
        self.max_metric = None          # Maximum value of metric that is still considered close to the nullspace

        self.nullspace_res_dict = {}    # Filling with results of the runs, nullspace vectors etc.
        self.results = pd.DataFrame()   # Filling with fresults from the run. Overview!

    def fit_nullspace(self, plot=0):

        for feat in self.feat_funcs: 
            pass
        return self 
    
    def plot_feat_nullspace_run(self): 
        return self

    def anlyse_features(self, color_dict, 
        cv=1, include_cv_model=0, opt_mape=0, opt_dist=1,
        scatter=0, plot_cv=0, max_mape=-0.5):
        ''' Function to anlayse features given certain data! (: '''

        # PLS 1 model is always selected as a reference. 
        # Models
        models = [
            PLSRegression(n_components=1, tol=1e-7, scale=False)
        ]
        model_names = [
            'PLS 1 comp'
        ]

        for fun in self.feat_fun:
            # Calculate the feature and linearized coef.
            x_hat, lin_coef_, lin_const_coef = self.regress_linearized_coeff(fun)


            # if scatter:
                # Make a scatterpliot here, to see whats going on.
                # Where do you linearize? The more nonlinear the more outlier, the crappier this mehtod will be!
                # Whats the path regression coefficeints take with varying regularization?
                # plt.scatter(X@lin_coef_.reshape(-1), y_gt[:, 1])
                # plt.scatter(np.array([fun_targetj[1](X[i, :]) for i in range(X.shape[0])]), y_gt[:, 1])
                # lin_sp = np.linspace(np.min(y_gt[:, 1]), np.max(y_gt[:, 1]), 10)
                # plt.plot(lin_sp, lin_sp)

            # CV only if feature is linear!
            if cv:
                cv_dict = optimise_pls_cv(
                    self.X_, self.y_, max_comps=10, plot_components=plot_cv, std=False, min_distance_search=True, featlin=lin_coef_)
                rmse_min_comp = cv_dict['components'][cv_dict['rmse_std_min']]
                if include_cv_model:
                    models.append(PLSRegression(n_components=rmse_min_comp, tol=1e-7, scale=False),)
                    model_names.append('PLS ' + str(rmse_min_comp) + ' comp')
                    print(model_names)

            if opt_mape:
                mape_lin = 100*mean_absolute_percentage_error(y, self.X@lin_coef_.reshape(-1))
                alpha = optimize_regcoef_mape('ridge', self.X_, self.y_, [10**5, 10**(-5)], mape_lin+1.5, max_depth=10)
                models.append(Ridge(alpha=alpha))
                model_names.append(f" RR: {alpha:.2f}")

            if opt_dist:
                alpha = optimize_regcoef_dist('ridge', self.X_, self.y_, [10**5, 10**(-5)], lin_coef_, norm=1, max_depth=10)
                # models.append(Ridge(alpha=alpha))
                # model_names.append(f" RR: {alpha:.2f}")

                comp = optimize_regcoef_dist('PLS', self.X_, self.y_, [10], lin_coef_, norm=1, max_depth=10)
                print('opt_dict')
                # Ensures that this is the last list item by removing previous identical entries. 
                if f"PLS {comp} comp" in model_names:
                    id = model_names.index(f"PLS {comp} comp")
                    model_names.remove(f"PLS {comp} comp")
                    print(id)
                    models.pop(id)
                    # models.remove(PLSRegression(n_components=comp, tol=1e-7, scale=False))
                    models.append(PLSRegression(n_components=comp, tol=1e-7, scale=False))
                    model_names.append(f"PLS {comp} comp")
                    
        
            model_dict = {'models': models, 'model_names': model_names}
            label_dict = {'xlabel': 'Voltage (V)'}
            print(model_dict['model_names'])

        return self

    def linearization_plot(self, target_fun_id):
        fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [7, 3, 3]}, figsize=(36,7))
        linearization_regeression_row_plots(
            self.X_, self.x, self.y_, fun_targetj, axs, cmap, model_dict, color_dict, label_dict,
            nullspace_corr=True, plot_nullspace_bool=False, max_mape=max_mape)
        # plt.show()
        return fig, axs


    def regress_linearized_coeff(self, fun):
        """Estimation of m and b via OLS regression.
        """
        x_hat = np.zeros(len(self.X_))
        a = np.mean(self.X_, axis=0)
        gradient = jacfwd(fun)

        for i in range(len(self.X_)):
            # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            x_hat[i] = fun(a) + np.dot((self.X_[i, :]-a), gradient(a))
        
        reg = LinearRegression(fit_intercept=True).fit(x_hat.reshape(-1, 1), self.y_)

        m = reg.coef_
        b = reg.intercept_
        linearized_coef = m * gradient(a)
        linearized_const_coef = m*fun(a) + b
        
        return x_hat, np.array(linearized_coef), np.array(linearized_const_coef)


def jax_moment(X, power): 
    """rewriting the sample moment without prefactor! 1/n
    operating on a single row of a matrix
    using jax impolemtations to allow for autodifferentiation
    """
    X_tilde = jnp.array(X) - jnp.mean(X)
    if len(X.shape)==2:
        shape = X.shape[1]
    else:
        shape = X.shape[0]
    return jnp.sum(jnp.power(X_tilde, power))/shape

def linearization_regeression_row_plots(
    X, x, y, fun, axs, cmap, model_dict, color_dict, label_dict,
    nullspace_corr=True, plot_nullspace_bool=False, max_mape=-0.5):
    """
    """

    model_names = model_dict['model_names']
    models = model_dict['models']
    color_list = color_dict['color_list']
    marker_list = color_dict['marker_list']
    x_label = label_dict['xlabel']
    # Mean
    x_hat, lin_coef_, lin_const_coef = regress_linearized_coeff(X, y, fun)
    mape_linfeat = 100*mean_absolute_percentage_error(y, X@(lin_coef_.reshape(-1)))
    axs[0].plot(x, lin_coef_.reshape(-1), label='Linearized Weights' +f" MAPE: {mape_linfeat:.2f} %", lw=2.5, color=color_list[0], marker=marker_list[0], markevery=(0, 30),  markersize=9)

    # PLS with one and two components is always a good idea to get a better understanding
    for j, model in enumerate(models):
        reg = model.fit(X-np.mean(X, axis=0), y-y.mean())
        mape_reg = 100*mean_absolute_percentage_error(y, X@(reg.coef_.reshape(-1)))

        axs[0].plot(
            x, reg.coef_.reshape(-1), label=f"{model_names[j]}, MAPE: {mape_reg:.2f} %", lw=2.5, 
            color=color_list[np.mod(j+1, len(color_list))], 
            marker=marker_list[np.mod(j+1, len(marker_list))], 
            markevery=(5*(j+1), 30), markersize=9)
    axs[0].set_ylabel(r'$\beta$')
    axs[0].set_xlabel(x_label)
    axs[0].set_xlim([2.0, 3.5])

    if nullspace_corr: 
        # Create Nullspace object
        data = BasicsData(X=X, x=x, y=y)
        # Train the model with the regression coeficients that shall be testes
        nulls_ = Nullspace(data)
        nulls_.learn_weights([models[-1]], [model_names[-1]])
        # do the nullspace stuff
        if plot_nullspace_bool:
            nulls_, fig, ax = nulls_.nullspace_correction(
                key_alpha=model_names[-1], w_alpha_name=model_names[-1], 
                w_beta = lin_coef_.reshape(-1), w_beta_name='Mean Weights', std=False, 
                plot_results=True, save_plot=0, max_mape=max_mape)
        else: 
            nulls_ = nulls_.nullspace_correction(
                key_alpha=model_names[-1], w_alpha_name=model_names[-1], 
                w_beta = lin_coef_.reshape(-1), w_beta_name='Mean Weights', std=False, 
                plot_results=False, save_plot=0, max_mape=max_mape)

        y2 = nulls_.nullsp['w_alpha']+nulls_.nullsp['v_'][-1,:]

        from src.nullspace import format_label
        label = format_label(nulls_.max_gamma, max_mape)
        
        # label=r'close to $\mathcal{\mathbf{N}}(X) xyz$'
        # print(label)
        axs[0].fill_between(x.reshape(-1), nulls_.nullsp['w_alpha'], y2=y2, color='darkgrey', zorder=-1, alpha=0.8, label=label)

        #axs[0].fill_between(
        #    x, nulls_.nullsp['w_alpha'], y2=y2, color='darkgrey', 
        #    zorder=-1, alpha=0.8, label=r'close to $\mathcal{\mathbf{N}}(X)$')
        axs[0].legend(loc=3)


    # Middle: Non-linearity check
    # How good is the linear approximation:

    # Calculate the linear feature for all the rows
    feat_nonlin = np.zeros(len(X))
    a = np.mean(X, axis=0)
    fun_a = fun(a)

    # Regress the linear

    for j in range(len(X)):
        # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        feat_nonlin[j] = fun(X[j, :])

    # Throw it all into a plotter
    plot_linearized_nonlinear_comp(
        feat_nonlin, x_hat, y, fun_a, cmap=cmap, 
        title='', xlabel='Linearized Feature', ylabel='Feature', ax=axs[1])

    # Right: Pearson correlation coefficient
    plot_pearson_corr_coef_comp(
        feat_nonlin,  y, cmap, title='Person Correlation', xlabel='Feature', ylabel='y', ax=axs[2])

    return axs

@mpl.rc_context(fname='./styles/linearization_plot.mplstyle')
def linearization_plots(x,  X, y, fun_targetj, fun_target_names, models, model_names, plot_labels, cmap=sns.color_palette("icefire", as_cmap=True), show=True):
    """ Function to create plot of data and regression coefficients
    x: 1d array for units on the x-axis
    X: Training data, 2D array
    y: Training labels, 2D array, where each of tjhe columsn corresponds to the respective data generating mechanism.
    model: list of models that come with a .fit() function and contain a .coef property
    Ideas for labeling taken from Joe Kingtons answer. 
    https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    """
    color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
    marker_list = ['s', 'o', 'D', 'P']
    rows = len(fun_targetj) + 2
    columns = 3
    figsize = [11*columns, 6*rows]

    row_labels = ['Data', 'Data Stats.']
    row_labels = row_labels + [fname +' Feature' for fname in fun_target_names]

    fig, axs = plt.subplots(rows, columns, gridspec_kw={'width_ratios': [7, 3, 3]}, figsize=figsize)

    # Think about whether to include plot of test set in here as well.
    # 0, 0: Data
    axs[0, 0] = plot_x_tt2(axs[0, 0], x, X, color_list[0], plot_labels['xdata_label'], plot_labels['ydata_label'])
    # 0, 1: Column correlations 
    axs[0, 1] = plot_corrheatmap(axs[0, 1], x, X, cmap, plot_labels['xdata_label'], plot_labels['xdata_label'], '|Corr.| Columns')
    # 0, 2: Row correlations
    axs[0, 2] = plot_corrheatmap(axs[0, 2], np.arange(X.T.shape[1]), X.T, cmap, plot_labels['row_label'], plot_labels['row_label'], '|Corr.| Rows', cols=False)
    # 1, 0: Mean & std of data
    axs[1, 0] =plot_stats(axs[1, 0], x, X, color_list[0], color_list[1], color_list[2], plot_labels['xdata_label'], plot_labels['ydata_label'])
    
    # Empty plots could be filled with correlation between the training and the test dataset rows. 
    # This would provide insights into issue with the data splitting
    # 1, 1: Empty
    axs[1, 1].axis('off')
    # 1, 2: Empty
    axs[1, 2].axis('off')

    # Row 2 and onwards
    # One row for every target function
    for i in range(len(fun_targetj)):
        # Left: Linearized regression coefficients
        # & learned regression coefficients
        try:
            y_train = y[:, i]
        except: 
            y_train = y

        model_dict = {'models': models, 'model_names': model_names}
        color_dict = {'color_list': color_list, 'marker_list': marker_list}
        label_dict = {'xlabel': 'Voltage (V)'}
        linearization_regeression_row_plots(
            X, x, y_train, fun_targetj[i], axs[i+2, :], cmap, model_dict, color_dict, label_dict,
            nullspace_corr=True, plot_nullspace=False)

    pad = 30 # in points

    for ax, row in zip(axs[:,0], row_labels):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=45)

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need 
    # to make some room. These numbers are are manually tweaked. 
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95, hspace = 0.2)

    if show: 
        plt.show()
            
    return fig, axs

def plot_linearized_nonlinear_comp(feature_non_lin, feature_linearized, y_train, 
                                   center_taylor, cmap,
                                   title='', xlabel='', ylabel='', ax=None):
    '''Plots the linear and non linear scalar feature values and makes a visual comparison!
    '''
    # Get axis object in case no axis object was passed explicitly
    # if ax is None:
    #    print('ax is None, create aixs')
    #    ax = plt.gca()
    y_train_norm = (y_train-y_train.min())/(y_train.max()-y_train.min())

    nrmse = mean_squared_error(feature_non_lin, feature_linearized, squared=False)/np.abs(feature_non_lin.max()-feature_non_lin.min())
    # ind_ = np.where((y>=350) & (y<=1500))
    # rmse_ = mean_squared_error(feature_non_lin[ind_], feature_linearized[ind_], squared=False)
    
    rss = np.sum(np.abs(feature_non_lin - feature_linearized))

    for i in range(len(feature_non_lin)):
        ax.scatter(feature_linearized[i], feature_non_lin[i], color=cmap(y_train_norm[i]), s=100)
    
    h1 = ax.scatter(center_taylor, center_taylor, marker="+", s=35**2, linewidths=3,
                    label=r'$\mathbf{a}=\overline{\mathbf{x}}^{\mathrm{train}}$')
    vals  = np.linspace(
        feature_linearized.min(),
        feature_linearized.max(), 10)
    
    ax.plot(vals, vals)
    
    cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)


    # How good is the linear approximation:
    mean_fx = np.mean(feature_non_lin)
    f_mean_x = center_taylor
    rel_deviation = np.abs((f_mean_x-mean_fx)/mean_fx)

    textstr = '\n'.join((
        'NRMSE: %.3f' % (nrmse),
        # 'RMSE Central Region: %.2f' % (rmse_),
        'Dev at center: %.2f' % (100*rel_deviation) + '%',
        ))
    h2 = ax.plot([], [], ' ', label=textstr)
    
    # Fix overlapping axis ticks in case of small numbers
    if np.abs(feature_linearized.max()-feature_linearized.min()) < 0.01:
        ax.ticklabel_format(axis='both', style='sci',  useOffset=True, scilimits=(0,0))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    order = [1,0]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.grid()

    return

def plot_pearson_corr_coef_comp(feature_non_lin,  y_train, cmap,
                                 title='Person Correlation', xlabel='', ylabel='', ax=None):
    '''Plots the feature values and y response values, display the person corr coeff
    '''
    # Get axis object in case no axis object was passed explicitly
    # if ax is None:
    #    print('ax is None, create aixs')
    #    ax = plt.gca()
 
    corr_coeff = np.corrcoef(np.column_stack((feature_non_lin, y_train)), rowvar=False)

    y_train_norm = (y_train-y_train.min())/(y_train.max()-y_train.min())
    for i in range(len(feature_non_lin)):
        ax.scatter(feature_non_lin[i], y_train[i], color=cmap(y_train_norm[i]), s=100)
    
    cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)

    h2 = ax.plot([], [], ' ', label='Pearson corr: %.2f' % corr_coeff[0, 1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # handles, labels = ax.get_legend_handles_labels()
    # order = [1,0]
    ax.legend()
    ax.grid()

    return