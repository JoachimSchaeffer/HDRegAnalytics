'''
Featue Linearization Methodology
Fuctions to linearize nonlinear features and 
Subsequently finding a constant term via regeression to match the metdoch
'''

# Packages
from matplotlib import markers
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

# ToDo: Write matplotlib wrapper similar to seaborn for the functions 
# we want to use here. (scatter plot including nolinearity measure and pearson correlation coeff)
from src.src_lin_feature import plot_linearized_nonlinear_comp
from src.src_lin_feature import plot_pearson_corr_coef_comp

from sklearn.linear_model import LinearRegression

import jax.numpy as jnp
from jax import grad
from jax import jacfwd


def regress_linearized_coeff(X_train, y_train, fun):
    '''Estimation of m and b via OLS regression.
    '''
    x_hat = np.zeros(len(X_train))
    a = np.mean(X_train, axis=0)
    gradient = jacfwd(fun)

    for i in range(len(X_train)):
        # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        x_hat[i] = fun(a) + np.dot((X_train[i, :]-a), gradient(a))
    
    reg = LinearRegression(fit_intercept=True).fit(x_hat.reshape(-1, 1), y_train-y_train.mean())

    m = reg.coef_
    b = reg.intercept_
    linearized_coef = m * gradient(a)
    linearized_const_coef = m*fun(a) + b
    
    return x_hat, np.array(linearized_coef), np.array(linearized_const_coef)

def jax_moment(X, power): 
    '''rewriting the sample moment without prefactor! 1/n
    operating on a single row of a matrix
    using jax impolemtations to allow for autodifferentiation
    '''
    X_tilde = jnp.array(X) - jnp.mean(X)
    if len(X.shape)==2:
        shape = X.shape[1]
    else:
        shape = X.shape[0]
    return jnp.sum(jnp.power(X_tilde, power))/shape

@mpl.rc_context(fname='./styles/linearization_plot.mplstyle')
def linearization_plots(x,  X, y, fun_targetj, fun_target_names, models, model_names, plot_labels, cmap=sns.color_palette("icefire", as_cmap=True), show=True):
    ''' Function to create plot of data and regression coefficients
    x: 1d array for units on the x-axis
    X: Training data, 2D array
    y: Training labels, 2D array, where each of tjhe columsn corresponds to the respective data generating mechanism.
    model: list of models that come with a .fit() function and contain a .coef property
    Ideas for labeling taken from Joe Kingtons answer. 
    https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    '''
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
    axs[0, 0].plot(x, X[:, :].T, label='Train', lw=1, color=color_list[0])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0, 0].legend(by_label.values(), by_label.keys(), loc=4)
    #axs[0, 0].set_title('Training Data')
    axs[0, 0].set_xlabel(plot_labels['xdata_label'])
    axs[0, 0].set_ylabel(plot_labels['ydata_label'])

    # 0, 1: Column correlations 
    # 0, 2: Row correlations
    X_df = pd.DataFrame(X[:, :])
    X_trans_df = pd.DataFrame(X[:, :].T)
    X_corr = X_df.corr()
    X_corr.set_index(np.round(x, 1), inplace=True)
    X_corr.set_axis(np.round(x, 1), axis='columns', inplace=True)
    mask = np.triu(X_corr)

    ax = sns.heatmap(
        X_corr, 
        vmin=-0.2, vmax=1, center=0.4,
        cmap=cmap,
        square=True, 
        xticklabels=100,
        yticklabels=100,
        ax = axs[0, 1],
        mask=mask
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    axs[0, 1].set_title('Corr. of Columns')
    axs[0, 1].set_xlabel(plot_labels['xdata_label'])
    axs[0, 1].set_ylabel(plot_labels['xdata_label'])
    # axs[0, 1].set_xticks(np.range(0, len(X_corr)), labels=range(2011, 2019))

    X_trans_corr = X_trans_df.corr()
    mask = np.triu(X_trans_corr)
    ax = sns.heatmap(
        X_trans_corr, 
        vmin=0.7, vmax=1, center=0.85,
        cmap=cmap,
        square=True,
        xticklabels=10,
        yticklabels=10,
        ax = axs[0, 2],
        mask=mask
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    axs[0, 2].set_title('Corr. of Rows')
    axs[0, 2].set_xlabel(plot_labels['row_label'])
    axs[0, 2].set_ylabel(plot_labels['row_label'])
    
    # 1, 0: Mean & std of data
    axs[1, 0].plot(x, np.abs(np.mean(X.T, axis=1)), label='|Mean Train|', lw=2.5, color=color_list[0])
    axs[1, 0].plot(x, np.std(X.T, axis=1), label='Std Train', lw=2.5, color=color_list[1])
    axs[1, 0].plot(x, np.abs(np.median(X.T, axis=1)), label='|Median Train|', lw=2.5, color=color_list[2])
    axs[1, 0].legend(loc=2)
    axs[1, 0].set_xlabel(plot_labels['xdata_label'])
    axs[1, 0].set_ylabel(plot_labels['ydata_label'])
    
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

        x_hat, lin_coef_, lin_const_coef = regress_linearized_coeff(X, y_train, fun_targetj[i])
        axs[i+2, 0].plot(x, lin_coef_.reshape(-1), label=r'$\beta_{Lin}$', lw=2.5, color=color_list[0], marker=marker_list[0], markevery=(0, 30),  markersize=9)
        for j, model in enumerate(models):
            reg = model.fit(X-np.mean(X, axis=0), y_train-y_train.mean())
            axs[i+2, 0].plot(
                x, reg.coef_.reshape(-1), label=model_names[j], lw=2.5, 
                color=color_list[np.mod(j+1, len(color_list))], 
                marker=marker_list[np.mod(j+1, len(marker_list))], 
                markevery=(5*(j+1), 30), markersize=9)
            
        axs[i+2, 0].legend(loc=2)
        axs[i+2, 0].set_ylabel(r'$\beta$')
        axs[i+2, 0].set_xlabel(plot_labels['xdata_label'])
        # axs[i+2, 0].set_title(fun_target_names[i]+' Feature')

        # Middle: Non-linearity check
        # How good is the linear approximation:

        # Calculate the linear feature for all the rows
        feat_nonlin = np.zeros(len(X))
        a = np.mean(X, axis=0)
        fun_a = fun_targetj[i](a)

        # Regress the linear

        for j in range(len(X)):
            # np.dot: If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            feat_nonlin[j] = fun_targetj[i](X[j, :])
        
        # Throw it all into a plotter
        plot_linearized_nonlinear_comp(feat_nonlin, x_hat, y_train, fun_a,
                                cmap=cmap, 
                                title=fun_target_names[i]+' Feature', xlabel='Linearized Feature', ylabel='Feature', 
                                ax=axs[i+2, 1])

        # Right: Pearson correlation coefficient
        plot_pearson_corr_coef_comp(feat_nonlin,  y_train, cmap,
                                 title='Person Correlation', xlabel='Feature', ylabel='y', ax=axs[i+2, 2])

    

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