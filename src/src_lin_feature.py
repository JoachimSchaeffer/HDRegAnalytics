# This source file is a collection of helper functions 
# needed to run the ``Linearized_feature_comparison'' jupyter notebook
# Some of them could potentially be intergrated in the oobject oriented structure
# for experimental reasons this was not done yet 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib import rc
# rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
# plt.style.use('plots-latex.mplstyle')
plt.style.use('./styles/plots.mplstyle')


from sklearn.metrics import mean_squared_error


def norm_minus1_1(w0):
    '''-1,1 normalization 
    '''
    return w0/w0.max()

def norm_0_1(w0):
    '''0,1 normalization 
    '''
    max_range_w0 = w0.max()-w0.min()  
    return (w0-w0.min())/max_range_w0

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''Truncates a colormap. This is important because a) many people are partly colorblind and a lot of 
    colormaps unsuited for them, and b) a lot of colormaps include yellow whichcanbe hard to see on some 
    screens and bad quality prints. 
    from https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    '''
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def color_map_rgb(x): 
    cmap = truncate_colormap(cm.get_cmap('plasma'), 0.05, 0.85)
    #Select the color 75% of the way through the colorscale   
    rgba = cmap(x)
    rgba = tuple(int((255*x)) for x in rgba[0:3])
    rgba = 'rgb'+str(rgba)
    return rgba

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

def check_linearity(fun, X, lin_a):
    '''Function to check the linearity approximation'''
    # How good is the linear approximation:
    non_lin_a = np.mean(fun(X))
    # print('The value of the of log(var(mean(DQ))) is' + f' {ln_var_a:.2f}')
    # print('The value of the of mean(log(var((DQ))) is' + f' {mlg_varX:.2f}')
    rel_deviation = np.abs((non_lin_a-lin_a))/non_lin_a
    # print('The the deviation from linearity at the center of the Taylor aproximation is thus  ' + f' {rel_deviation*100:.2f}%')
    return rel_deviation

def plot_linear_model_weights_and_features(vector_lin, reg_coef, X,y_train,  Vdlin,\
                                           label_lin='', label_reg='', save=0, path=''):
  
    fig, axs = plt.subplots(2,1, figsize=[12,12], sharex=True, sharey=True)
    
    colors = ['#0051a2', '#97964a', '#f4777f', '#93003a']
    #colors = ['#00429d', '#628ebd', '#86b6ca', '#b1ded3', '#ffffcb']
    for i in range(len(vector_lin)):
        axs[0].plot(Vdlin, vector_lin[i], label=label_lin[i], color=colors[i], lw=2.5)
    
    for j in range(len(reg_coef)):
        axs[0].plot(Vdlin, reg_coef[j], label=label_reg[j], color=colors[len(vector_lin)+j], lw=2.5)

    reds_ = plt.get_cmap('Reds_r')
    reds = truncate_colormap(reds_, 0, 0.7)
    cNorm  = mcolors.Normalize(vmin=0, vmax=3000)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=reds)

    idx_median = np.where(y_train==np.median(y_train))[0][0] 
    idx_max = np.where(y_train==np.max(y_train))[0][0] 
    idx_min = np.where(y_train==np.min(y_train))[0][0] 
    QD_100_10_median_meansub = X[idx_median,:]-np.mean(X, axis=0)
    QD_100_10_max_meansub = X[idx_max,:]-np.mean(X, axis=0)
    QD_100_10_min_meansub = X[idx_min,:]-np.mean(X, axis=0) 
    
    pqd2, = axs[1].plot(Vdlin, QD_100_10_max_meansub, color=scalarMap.to_rgba(2160), 
                     label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ High Cycle Life', lw=2.5)
    pqd, = axs[1].plot(Vdlin, QD_100_10_median_meansub, color=scalarMap.to_rgba(10**np.median(y_train)),
                     label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ Medium Cycle Life', lw=2.5)
    pqd1, = axs[1].plot(Vdlin, QD_100_10_min_meansub, color=scalarMap.to_rgba(300), 
                     label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ Low Cycle Life', lw=2.5)

    axs[1].set_xlabel(r'Voltage (V)')
    axs[0].set_ylabel(r'Regression Coefficients')
    axs[1].set_ylabel(r'$\Delta \mathbf{Q}_{100\mathrm{-}10} - \overline{\Delta \mathbf{Q}}_{100\mathrm{-}10}^{train}$')      
    axs[1].set_xlim([2.0,3.5])
    
    axs[0].grid()
    axs[1].grid()
    
    axs[0].legend(loc=3)
    axs[1].legend(loc=3)

    plt.tight_layout()
    if save:
        plt.savefig(path + 'Coeff_Subplot_PLSTayloretc.pdf', bbox_inches='tight', dpi=300)
        
    return None

def plot_linear_model_weights_and_features_old(vector_lin, reg_coef, X,y_train,  Vdlin,\
                                           ylim=[-0.14, 0.08], label_lin='', \
                                           label_reg='', loc_legend=3, save=0, path=''):
  
    fig, host = plt.subplots()
    par1 = host.twinx()

    p1, = host.plot(Vdlin, vector_lin, color='k', label=label_lin)
    p2, = host.plot(Vdlin, reg_coef, color='C0', label=label_reg)

    reds_ = plt.get_cmap('Reds_r')
    reds = truncate_colormap(reds_, 0, 0.7)
    cNorm  = mcolors.Normalize(vmin=0, vmax=3000)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=reds)

    idx_median = np.where(y_train==np.median(y_train))[0][0] 
    idx_max = np.where(y_train==np.max(y_train))[0][0] 
    idx_min = np.where(y_train==np.min(y_train))[0][0] 
    QD_100_10_median_meansub = X[idx_median,:]-np.mean(X, axis=0)
    QD_100_10_max_meansub = X[idx_max,:]-np.mean(X, axis=0)
    QD_100_10_min_meansub = X[idx_min,:]-np.mean(X, axis=0) 

    pqd, = par1.plot(Vdlin, QD_100_10_median_meansub, color=scalarMap.to_rgba(np.e**np.median(y_train)),
                     label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ Medium Cycle Life')
    pqd1, = par1.plot(Vdlin, QD_100_10_min_meansub, color=scalarMap.to_rgba(300), 
                      label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ Low Cycle Life')
    pqd2, = par1.plot(Vdlin, QD_100_10_max_meansub, color=scalarMap.to_rgba(2160), 
                      label=r'$\Delta \mathbf{Q}_{100\mathrm{-}10}$ High Cycle Life')

    cb = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=2160, clip=False), cmap=reds), 
                      ax=par1, pad=0.15)
    cb.set_label('Cycle Life', labelpad=10)

    host.set_xlabel(r'Voltage (V)')
    host.set_ylabel(r'Regression Coefficients')
    par1.set_ylabel(r'$\Delta \mathbf{Q}_{100\mathrm{-}10} - \overline{\Delta \mathbf{Q}}_{100\mathrm{-}10}^{train}$')      

    host.set_ylim(ylim)
    par1.set_ylim(ylim)
    
    par1.yaxis.label.set_color(pqd.get_color())
    par1.tick_params(axis='y', colors=pqd.get_color())
    host.grid()
    lines = [p1, p2, pqd, pqd1, pqd2]
    host.set_xlim([2.0,3.5])

    legend = par1.legend(lines, [l.get_label() for l in lines], loc=loc_legend)

    if save:
        plt.savefig(path + label_lin[-12:].replace(' ','') + 'PLSTaylor.pdf', bbox_inches='tight', dpi=300)

    plt.show()
    return None

def plot_prediction_results(y, y_pred, base=10, title=None, save=False, 
                            return_ax=0, ax=None, path='',logy=1):
    rc('text', usetex=False)
    plt.style.use('./styles/plots.mplstyle')
    if ax==None:
        fig, ax = plt.subplots()
        
    if len(y.keys())<=2:
        labels = ['Training', 
                  'Test',
                 ]
    elif len(y.keys())==3:
        labels = ['Training',
                  'Prim. Test',
                  'Sec. Test',
                 ]
    else: 
        raise ValueError('Only two test sets allowed. Check lenght of y dict')
        
    markers = ['o', '^', 's']
    colors = ['b', 'C1', 'r']
    if(logy):
        for i, key in enumerate(y.keys()):
            ax.scatter(base**y[key], base**y_pred[key], marker=markers[i], color=colors[i], label=labels[i])
    else:
        for i, key in enumerate(y.keys()):
            ax.scatter(y[key], y_pred[key], marker=markers[i], color=colors[i], label=labels[i])
            
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 2500)
    ax.plot([0,2500], [0,2500], color='k')
    ax.set_aspect('equal', 'box')

    ax.set_xlabel('Observed Cycle Life')
    ax.set_ylabel('Predicted Cylce Life')
    
    if title != None:
        ax.set_title(str(title))
    if(logy):
        RMSE_dict = calc_rmse_dict(y, y_pred, base=base)
        RMSE_dict_low = calc_rmse_dict(y, y_pred, base=base, up_lim=1500)        
    else:
        RMSE_dict = calc_rmse_dict(y, y_pred, base=base,logy=0)
        RMSE_dict_low = calc_rmse_dict(y, y_pred, base=base, up_lim=1500,logy=0)     


    
    if len(y.keys())==1:
        textstr = '\n'.join((
        'RMSE:', 
        '  Training = %.1f' % (RMSE_dict['train'], ),
        'RMSE < 1500:',
        '  Training = %.1f' % (RMSE_dict_low['train'], ),
            )) 
    elif len(y.keys())==2:
        textstr = '\n'.join((
        'RMSE:', 
        '  Training = %.1f' % (RMSE_dict['train'], ),
        '  Test = %.1f' % (RMSE_dict['test'], ),
        'RMSE < 1500:',
        '  Training = %.1f' % (RMSE_dict_low['train'], ),
        '  Test = %.1f' % (RMSE_dict_low['test'], ),
            ))
    elif len(y.keys())==3:
        textstr = '\n'.join((
        'RMSE:', 
        '  Training = %.1f' % (RMSE_dict['train'], ),
        '  Prim. Test = %.1f' % (RMSE_dict['test'], ),
        '  Sec. Test = %.1f' % (RMSE_dict['test2'], ),
        'RMSE < 1500:',
        '  Training = %.1f' % (RMSE_dict_low['train'], ),
        '  Prim. Test = %.1f' % (RMSE_dict_low['test'], ),
        '  Sec. Test = %.1f' % (RMSE_dict_low['test2'], ),
            ))
    else:
        raise ValueError('Only two test sets allowed. Check lenght of y dict')
            
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        verticalalignment='top', bbox=props)
        
    ax.legend(loc='upper left', bbox_to_anchor=(0.4, 0., 0.6, 1))
    
    if save: 
        plt.tight_layout()
        plt.savefig(path + str(title).replace(" ", "_") + '.pdf', 
                    bbox_inches='tight', dpi=300)
    
    if return_ax: 
        return ax
    
    else:
        plt.show()
        return None 

    
def plot_nl_results(y, y_pred, base=10, title=None, save=False, 
                            return_ax=0, ax=None, path='',logy=1):
    rc('text', usetex=False)
    plt.style.use('./styles/plots.mplstyle')
    if ax==None:
        fig, ax = plt.subplots()
        
    if len(y.keys())<=2:
        labels = ['Training', 
                  'Test',
                 ]
    elif len(y.keys())==3:
        labels = ['Training',
                  'Prim. Test',
                  'Sec. Test',
                 ]
    else: 
        raise ValueError('Only two test sets allowed. Check lenght of y dict')
        
    markers = ['o', '^', 's']
    colors = ['b', 'C1', 'r']
    if(logy):
        for i, key in enumerate(y.keys()):
            ax.scatter(base**y[key], base**y_pred[key], marker=markers[i], color=colors[i], label=labels[i])
    else:
        for i, key in enumerate(y.keys()):
            ax.scatter(y[key], y_pred[key], marker=markers[i], color=colors[i], label=labels[i])
            
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 2500)
    ax.plot([0,2500], [0,2500], color='k')
    ax.set_aspect('equal', 'box')

    ax.set_xlabel('Observed Cycle Life')
    ax.set_ylabel('Predicted Cylce Life')
    
    if title != None:
        ax.set_title(str(title))
    if(logy):
        RMSE_dict = calc_NL_Ass_dict(y, y_pred, base=base)
        RMSE_dict_low = calc_NL_Ass_dict(y, y_pred, base=base, up_lim=1500)        
    else:
        RMSE_dict = calc_NL_Ass_dict(y, y_pred, base=base,logy=0)
        RMSE_dict_low = calc_NL_Ass_dict(y, y_pred, base=base, up_lim=1500,logy=0)     


    
    if len(y.keys())==1:
        textstr = '\n'.join((
        'RMSE NL-Ass:', 
        '  Training = %.4f' % (RMSE_dict['train'], ),
        'RMSE NL-Ass < 1500:',
        '  Training = %.4f' % (RMSE_dict_low['train'], ),
            )) 
    elif len(y.keys())==2:
        textstr = '\n'.join((
        'RMSE NL-Ass:', 
        '  Training = %.4f' % (RMSE_dict['train'], ),
        '  Test = %.4f' % (RMSE_dict['test'], ),
        'RMSE NL-Ass < 1500:',
        '  Training = %.4f' % (RMSE_dict_low['train'], ),
        '  Test = %.4f' % (RMSE_dict_low['test'], ),
            ))
    elif len(y.keys())==3:
        textstr = '\n'.join((
        'RMSE NL-Ass:', 
        '  Training = %.4f' % (RMSE_dict['train'], ),
        '  Prim. Test = %.4f' % (RMSE_dict['test'], ),
        '  Sec. Test = %.4f' % (RMSE_dict['test2'], ),
        'RMSE NL-Ass< 1500:',
        '  Training = %.4f' % (RMSE_dict_low['train'], ),
        '  Prim. Test = %.4f' % (RMSE_dict_low['test'], ),
        '  Sec. Test = %.4f' % (RMSE_dict_low['test2'], ),
            ))
    else:
        raise ValueError('Only two test sets allowed. Check lenght of y dict')
            
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        verticalalignment='top', bbox=props)
        
    ax.legend(loc='upper left', bbox_to_anchor=(0.4, 0., 0.6, 1))
    
    if save: 
        plt.tight_layout()
        plt.savefig(path + str(title).replace(" ", "_") + '.png', 
                    bbox_inches='tight', dpi=300)
    
    if return_ax: 
        return ax
    
    else:
        plt.show()
        return None 
    
def calc_rmse_dict(y, y_pred, base=10, up_lim=np.inf, l_lim=-np.inf,logy=1):
    RMSE_dict = dict.fromkeys(y.keys())
    if(logy):
        for key in y.keys():
            ind = (np.power(base, y[key])<=up_lim) & (np.power(base, y[key])>=l_lim)
            RMSE_dict[key] = mean_squared_error(np.power(base, y[key][ind]), np.power(base, y_pred[key][ind]), squared=False)
    else:
        for key in y.keys():
            ind = ((y[key])<=up_lim) & (y[key]>=l_lim)
            RMSE_dict[key] = mean_squared_error(y[key][ind], y_pred[key][ind], squared=False)
        
    return RMSE_dict

def calc_NL_Ass_dict(y, y_pred, base=10, up_lim=np.inf, l_lim=-np.inf,logy=1):
    RMSE_dict = dict.fromkeys(y.keys())
    if(logy):
        for key in y.keys():
            ind = (np.power(base, y[key])<=up_lim) & (np.power(base, y[key])>=l_lim)
            RMSE_dict[key] = mean_squared_error(np.power(base, y[key][ind]), np.power(base, y_pred[key][ind]), squared=False) /                         np.linalg.norm(np.power(base, y[key][ind]), ord=2) 
    else:
        for key in y.keys():
            ind = ((y[key])<=up_lim) & (y[key]>=l_lim)
            RMSE_dict[key] = mean_squared_error(y[key][ind], y_pred[key][ind], squared=False) / np.linalg.norm(y[key][ind], ord=2)
        
    return RMSE_dict