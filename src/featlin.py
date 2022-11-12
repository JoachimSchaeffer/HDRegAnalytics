"""
Featue Linearization Methodology
Fuctions to linearize nonlinear features and 
Subsequently finding a constant term via regeression to match the metdoch
"""

# Packages
from audioop import mul
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.colors as clr

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import copy

from src.basis import BasicsData
from src.nullspace import Nullspace

from src.helper import optimize_cv
from src.helper import optimize_regcoef_dist

import jax.numpy as jnp
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
            self.feat_fun_dict = {feat_fun_names[i] : feat_fun[i] for i in range(len(feat_fun))}
        else:
            self.feat_fun_dict = feat_funcs

        self.nullspace_dict = dict.fromkeys(self.feat_fun_dict.keys())    # Filling with results of the runs, nullspace vectors etc.
        columns = ['Featue', 'Model', 'CV', 'min_dist', 'NRMSE', 'Pearson']
        self.results = pd.DataFrame(columns=columns)   # Filling with fresults from the run. Overview!

        # Making plotting stuff a lot easier by setting some color combinations
        # colors = ['#332bb3', '#4a31b5', '#5d37b6', '#6d3db7', '#7c43b7', '#8a49b6', '#964fb5', '#a256b3', '#ad5db1', '#b764b0', '#c16cae', '#ca75ad', '#d27eac', '#d989ab', '#e094aa', '#e7a1ab', '#ecafac', '#f0beae', '#f4cfb0', '#f6e1b4']
        colors = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']
        colors_IBM = ['#648fff', '#785ef0', '#dc267f', '#fe6100', '#ffb000',  '#000000']
        self.cmap_ = clr.LinearSegmentedColormap.from_list('Blue-light cb-safe', colors, N=256)
        self.cmap = clr.LinearSegmentedColormap.from_list('Blue-light cb-IBM', colors_IBM[:-1], N=256)
        # color_list = ['#0051a2', '#97964a', '#f4777f', '#93003a']
        # self.color_list = [colors_IBM[0], colors_IBM[2], colors_IBM[3], colors_IBM[4], colors_IBM[5]]
        self.color_list = colors[::-1] + colors
        self.marker_list = ['s', 'o', 'D', 'P']

    def regress_linearized_coeff(self, fun, std=False):
        """Estimation of m and b via OLS regression.
        """
        #if std: 
        #    X = self.data.X_std
        #else: 
        #    X = self.data.X
        X = self.data.X
        y = self.data.y

        x_hat = np.zeros(len(X))
        a = np.mean(X, axis=0)
        gradient = jacfwd(fun)

        for i in range(len(X)):
            # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            x_hat[i] = fun(a) + np.dot((X[i, :]-a), gradient(a))
        
        reg = LinearRegression(fit_intercept=True).fit(x_hat.reshape(-1, 1), y)

        m = reg.coef_
        b = reg.intercept_
        linearized_coef = m * gradient(a)
        # The constant coefficient should be equal to the mean of y
        linearized_const_coef = m*fun(a) + b
        # Test whether the constant coefficient is equal to the mean of y with a tolerance of 1% of the mean of y
        assert np.isclose(linearized_const_coef, np.mean(y), rtol=0.01), \
            f"Linearized constant coefficient is not equal to the mean of y \
              with a tolerance of 1% of the mean of y. Linearized constant coefficient: \
              {linearized_const_coef}, mean of y: {np.mean(y)}"
        
        if std:
            lin_coef = np.array(linearized_coef)*np.std(X, axis=0)
        else:
            lin_coef = np.array(linearized_coef)

        return x_hat, lin_coef, np.array(linearized_const_coef)
    
    def analyse_all_features(
        self, 
        opt_cv={'active':True, 'max_comp':10}, 
        opt_dist={'active':False}, max_nrmse=1, 
        fig_props={'save':False, 'multiple_fig': True}, std=False):
        """Analyses all Features"""

        if not fig_props['multiple_fig']: 
            fig = plt.figure(constrained_layout=True, figsize=(36,7*len(self.nullspace_dict.keys())))
            subfigs = fig.subfigures(nrows=len(self.nullspace_dict.keys()), ncols=1)
            #fig, axs = plt.subplots(len(self.nullspace_dict.keys()), 3, gridspec_kw={'width_ratios': [8, 2.5, 2.5]}, figsize=(36,7*len(self.nullspace_dict.keys()))
            #    )

        for i, key in enumerate(self.nullspace_dict.keys()): 
            self.analyse_feature(key, opt_cv=opt_cv, opt_dist=opt_dist, plot_cv=0, max_nrmse=max_nrmse, std=0)
            if fig_props['multiple_fig']:
                fig, ax = self.linearization_plot(key)
                if fig_props['save']: 
                    fig.suptitle(f'Linearized {key} Feature {fig_props["response"]}', y=0.94)
                    ax[0].set_xlabel(fig_props['ax0_xlabel'])
                    plt.tight_layout()
                    fig.savefig(fig_props['save_path'] + key + fig_props["response"] + '.pdf')
            else: 
                subfigs[i].suptitle(f'Linearized {key} Feature {fig_props["response"]}')
                axs = subfigs[i].subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [8, 2.5, 2.5]})
                fig, axs = self.linearization_plot(key, axs=axs, fig=fig)
        if not fig_props['multiple_fig']: 
            axs[0].set_xlabel(fig_props['ax0_xlabel'])
            # plt.tight_layout()
            fig.savefig(fig_props['save_path'] + 'LinerizationSummary' + fig_props["response"] + '.pdf')
        
        return self

    def analyse_feature(self, feat_key, std=False,
        opt_cv={'active':True, 'model': []}, opt_dist={'active':True, 'model': ['PLS', 'RR']},
        plot_cv=0, max_nrmse=1):
        ''' Function to anlayse features given certain data! (: '''

        # PLS 1 model is always selected as a reference. 
        # Models
        models = [
            PLSRegression(n_components=2, tol=1e-7, scale=False)
        ]
        model_names = [
            'PLS 2 comp'
        ]
        # List of results, should be in the same order as the pandas dataframe.
        results = []
        results.append(feat_key)
        if std:
            X = self.data.X_std
        else: 
            X = self.data.X_
        y = self.data.y_

        # Calculate the feature and linearized coef.
        x_hat, lin_coef_, lin_const_coef = self.regress_linearized_coeff(self.feat_fun_dict[feat_key], std=std)
        nrmse_linfeat = 100*mean_squared_error(y, X@(lin_coef_.reshape(-1)), squared=False)/(np.max(y)-np.min(y))
        
        # These if statements could be improved, for speed, but it works for now
        cv_dict_pls = optimize_cv(
            X, y, max_comps=10, alpha_lim=[10e-5, 10e3], folds=10, nb_stds=1, algorithm='PLS',
            plot_components=plot_cv, std=False, min_distance_search=opt_dist['active'], featlin=lin_coef_)
        cv_dict_rr = optimize_cv(
            X, y, max_comps=10, alpha_lim=[10e-5, 10e3], folds=10, nb_stds=1, algorithm='RR',
            plot_components=plot_cv, std=False, min_distance_search=opt_dist['active'], featlin=lin_coef_)

        if 'PLS' in opt_cv['model']:
            rmse_min_comp = cv_dict_pls['cv_res']['rmse_min_param']
            models.append(PLSRegression(n_components=rmse_min_comp, tol=1e-7, scale=False))
            model_names.append('PLS ' + str(rmse_min_comp) + ' comp')

        if 'PLS' in opt_dist['model']:
            comp = cv_dict_pls['l2_distance_res']['l2_min_param']
            # Legacy: 
            # comp = optimize_regcoef_dist('PLS', X, y, [10], lin_coef_, norm=opt_dist['norm'], max_depth=10)
            # Ensures that this is the last list item by removing previous identical entries. 
            if f"PLS {comp} comp" in model_names:
                id = model_names.index(f"PLS {comp} comp")
                model_names.remove(f"PLS {comp} comp")
                print(f"popping model with {id}")
                models.pop(id)
                # models.remove(PLSRegression(n_components=comp, tol=1e-7, scale=False))
            models.append(PLSRegression(n_components=comp, tol=1e-7, scale=False))
            model_names.append(f"PLS {comp} comp")

        if 'RR' in opt_cv['model']:
            alpha_min_rmse = cv_dict_rr['cv_res']['rmse_min_param']
            models.append(Ridge(alpha=alpha_min_rmse))
            model_names.append('RR ' + str(alpha_min_rmse))

        if 'RR' in opt_dist['model']:
            alpha = cv_dict_rr['l2_distance_res']['l2_min_param']
            # Legacy:
            # alpha = optimize_regcoef_dist('ridge', X, y, [10**5, 10**(-5)], lin_coef_, norm=opt_dist['norm'], max_depth=10)
            models.append(Ridge(alpha=alpha))
            model_names.append(f"RR: {alpha:.5f}")
    
        model_names.append('lfun')
        self.nullspace_dict[feat_key] = dict.fromkeys(model_names)
        self.nullspace_dict[feat_key]['lfun'] = dict.fromkeys(['feature_fun', 'lin_coef', 'nrmse'])
        self.nullspace_dict[feat_key]['lfun']['feature_fun'] = self.feat_fun_dict[feat_key]
        self.nullspace_dict[feat_key]['lfun']['lin_coef'] = lin_coef_
        self.nullspace_dict[feat_key]['lfun']['nrmse'] = nrmse_linfeat
        self.nullspace_dict[feat_key]['lfun']['x_hat'] = x_hat
        print(model_names)

        # Analyze the nullspace of the feature function
        # BROKEN ATM! for some reason the models are not added...
        for i, model in enumerate(models):
            results.append(model_names[i])
            reg = model.fit(X, y)
            nrmse_reg = 100*mean_squared_error(y, X@(reg.coef_.reshape(-1)), squared=False)/(np.max(y)-np.min(y))
            
            self.nullspace_dict[feat_key][model_names[i]] = dict.fromkeys(['model', 'nrmse', 'nulls_label', 'nulls'])
            # Not nice, but it is reliable.
            self.nullspace_dict[feat_key][model_names[i]]['model'] = copy.deepcopy(reg)
            self.nullspace_dict[feat_key][model_names[i]]['nrmse'] = nrmse_reg

            # Create Nullspace object
            # Train the model with the regression coeficients that shall be testes
            nulls_ = Nullspace(self.data)
            if std: 
                nulls_.std = True
            nulls_.learn_weights([model], [model_names[i]])
            nulls_ = nulls_.nullspace_correction(
                key_alpha=model_names[i], w_alpha_name=model_names[i], 
                w_beta = lin_coef_.reshape(-1), w_beta_name='', std=std, 
                plot_results=False, save_plot=0, max_nrmse=max_nrmse)

            self.nullspace_dict[feat_key][model_names[i]]['nulls'] = nulls_
            from src.nullspace import format_label
            label = format_label(nulls_.max_gamma, nulls_.max_nrmse)
            self.nullspace_dict[feat_key][model_names[i]]['nulls_label'] = label
        

        # Call function to transfer information in a result table.
        # self.label_dict = {'xlabel': 'Voltage (V)'}
        # print(self.model_dict['model_names'])

        return self

    def linearization_plot(self, feat_key, fig=None, axs=None):
        """Calls the linearization plotting function, resulting in a row of three plots for one feature"""
        if axs is None:
            fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [5.5, 2.5, 2.5]}, figsize=(24,5.3))
        axs = self.linearization_regeression_row_plots(feat_key, axs)
        # plt.show()
        return fig, axs

    def linearization_regeression_row_plots(self, feat_key, axs, label_dict={'xlabel' :'Voltage (V)'}): 
        """Plot a row of 3 subplots. 
        1: Regression coefficients 
        2: Linearization Analysis
        3: Pearson correlation coefficient
        """
        x_label = label_dict['xlabel']
        lin_coef_ = self.nullspace_dict[feat_key]['lfun']['lin_coef']

        # Linearized coefficients
        nrmse_linfeat = self.nullspace_dict[feat_key]['lfun']['nrmse']
        axs[0].plot(
            self.data.x, lin_coef_.reshape(-1), 
            label='Linearized Weights' +f" NRMSE: {nrmse_linfeat:.2f}%", color='k')
            # marker=self.marker_list[0], markevery=(0, 30),  markersize=9)

        # Other regression coefficients

        keys_models = list(self.nullspace_dict[feat_key].keys())
        model_names = [keys_models[i] for i in range(len(keys_models)) if keys_models[i]!='lfun']
        for j, model_name in enumerate(model_names):
            reg = self.nullspace_dict[feat_key][model_name]['model']
            nrmse_reg = self.nullspace_dict[feat_key][model_name]['nrmse']
            # nrmse_models = 100*mean_squared_error(self.data.X_std@(lin_coef_.reshape(-1)), self.data.X_std@(reg.coef_.reshape(-1)), squared=False)/(np.max(self.data.y_)-np.min(self.data.y_))
            # \n NRMSE(PLS, Lin): {nrmse_models}"
            if 'PLS' in model_name:
                marker = 's'
                color = '#ddcc77'
            if 'RR' in model_name:
                marker = 'o'
                color = '#117733'
            else:
                marker = 's'
            axs[0].plot(
                self.data.x, reg.coef_.reshape(-1), label=f"{model_names[j]}, NRMSE: {nrmse_reg:.2f}%", lw=2.5, 
                color=color, 
                marker=marker, markevery=(5*(j+1), 65), markersize=9)
        axs[0].set_ylabel(r'$\beta$')
        axs[0].set_xlabel(x_label)
        axs[0].set_xlim([2.0, 3.5])
        label = self.nullspace_dict[feat_key][model_name]['nulls_label']
        
        #label=r'close to $\mathcal{\mathbf{N}}(X) xyz$'
        #print(label)
        nulls_ = self.nullspace_dict[feat_key][model_name]['nulls']
        y2 = nulls_.nullsp['w_alpha']+nulls_.nullsp['v_'][-1,:]
        x = self.data.x
        axs[0].fill_between(x.reshape(-1), nulls_.nullsp['w_alpha'], y2=y2, color='darkgrey', zorder=-1, alpha=0.8, label=label)

        #axs[0].fill_between(
        #    x, nulls_.nullsp['w_alpha'], y2=y2, color='darkgrey', 
        #    zorder=-1, alpha=0.8, label=r'close to $\mathcal{\mathbf{N}}(X)$')
        axs[0].legend(loc='best', frameon=False)

        # Middle: Non-linearity check
        # How good is the linear approximation:
        X = self.data.X
        feat_nonlin = np.zeros(len(X))
        a = np.mean(X, axis=0)
        fun = self.nullspace_dict[feat_key]['lfun']['feature_fun']
        fun_a = fun(a)

        for j in range(len(X)):
            feat_nonlin[j] = fun(X[j, :])

        # Throw it all into a plotter
        x_hat = self.nullspace_dict[feat_key]['lfun']['x_hat']
        
        plot_linearized_nonlinear_comp(
            feat_nonlin, x_hat, self.data.y_, fun_a, cmap=self.cmap, 
            title='', xlabel='Linearized Feature', ylabel='Feature', ax=axs[1])

        # Right: Pearson correlation coefficient
        plot_pearson_corr_coef_comp(
            feat_nonlin,  self.data.y_, self.cmap, title='Person Correlation', xlabel='Feature', ylabel='y', ax=axs[2])

        return axs


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

    nrmse = mean_squared_error(feature_non_lin, feature_linearized, squared=False)/(np.max(feature_non_lin)-np.min(feature_non_lin))
    # ind_ = np.where((y>=350) & (y<=1500))
    # rmse_ = mean_squared_error(feature_non_lin[ind_], feature_linearized[ind_], squared=False)
    
    rss = np.sum(np.abs(feature_non_lin - feature_linearized))

    for i in range(len(feature_non_lin)):
        ax.scatter(feature_linearized[i], feature_non_lin[i], color='k', s=120) #color=cmap(y_train_norm[i]),
    
    h1 = ax.scatter(center_taylor, center_taylor, marker="+", s=35**2, linewidths=3,
                    label=r'$\mathbf{a}=\overline{\mathbf{x}}$')
    vals  = np.linspace(
        feature_linearized.min(),
        feature_linearized.max(), 10)
    
    ax.plot(vals, vals)
    
    # cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)


    # How good is the linear approximation:
    # mean_fx = np.mean(feature_non_lin)
    # f_mean_x = center_taylor
    # rel_deviation = np.abs((f_mean_x-mean_fx)/mean_fx)

    #textstr = '\n'.join((
    #    'NRMSE: %.3f' % (nrmse),
        # 'RMSE Central Region: %.2f' % (rmse_),
        # 'Dev at center: %.2f' % (100*rel_deviation) + '%',
    #    ))
    #h2 = ax.plot([], [], ' ', label=textstr)
    
    # Fix overlapping axis ticks in case of small numbers
    if np.abs(feature_linearized.max()-feature_linearized.min()) < 0.01:
        ax.ticklabel_format(axis='both', style='sci',  useOffset=True, scilimits=(0,0))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    order = [1,0]
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.legend(loc='best', frameon=False)
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
        ax.scatter(feature_non_lin[i], y_train[i], s=120, color='k') # color=cmap(y_train_norm[i]), s=100)
    
    # cb = plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max(), clip=False), cmap=cmap), ax=ax)
    # cb.set_label('Cycle Life', labelpad=10)

    h2 = ax.plot([], [], ' ', label=fr'$\rho=${corr_coeff[0, 1]:.2f}')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # handles, labels = ax.get_legend_handles_labels()
    # order = [1,0]
    ax.legend(loc='best', frameon=False)
    ax.grid()

    return