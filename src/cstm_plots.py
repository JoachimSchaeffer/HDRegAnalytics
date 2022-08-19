# Custom plotting functions to analyze the nature enrgy dataset in more detail

import numpy as np

from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d, Axes3D

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from scipy.spatial import ConvexHull

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from collections import defaultdict

import itertools
from src_lin_feature import color_map_rgb

def extract_Qdlin(bat_dict, keys, max_cycle, smooth=False):
    if smooth:
        with open('../../01Data/SmoothedDQNE/smnoothdq_dict10_100_full.pickle', 'rb') as handle:
            smooth_bat_raw_dict = pickle.load(handle)
        dataset = np.zeros((len(keys), 1000, max_cycle))
        for i, j in enumerate(keys):
            for k in range(8):
                dataset[i, :, k+1] = bat_dict[str(j)]['cycles'][str(k+1)]['Qdlin']
            for k in range(91):
                dataset[i, :, 8+k+1] = smooth_bat_raw_dict[str(j)][:,k]
            for k in range(max_cycle-1-99):
                dataset[i, :, 99+k+1] = bat_dict[str(j)]['cycles'][str(k+1)]['Qdlin']
    else: 
        dataset = np.zeros((len(keys), 1000, max_cycle))
        for i, j in enumerate(keys):
            for k in range(max_cycle-1):
                    dataset[i, :, k+1] = bat_dict[str(j)]['cycles'][str(k+1)]['Qdlin'] 
    return dataset


def extract_cyclelife(bat_dict, keys):
    dataset = np.zeros(len(keys))
                       
    for i, j in enumerate(keys):
        dataset[i] = bat_dict[str(j)]['cycle_life']
                       
    return dataset


def extract_charge_policy(bat_dict, keys):
    dataset = [None] * len(keys)
                      
    for i, j in enumerate(keys):
        dataset[i] = bat_dict[str(j)]['charge_policy']
    dataset = [s.split('-new')[0] for s in dataset]                  
    return dataset


def extract_cp_coord(cp_string):
    """
    pass a list of strings representaing the charging protocolls to this function
    return 3 coordinates by aplitting the string into individual sections
    """
    x1 = [s.split("C")[0] for s in cp_string]
    x2 = [s.split("-")[1] for s in cp_string]
    x2 = [s.split("C")[0] for s in x2]
    x3 = [s.split("(")[1] for s in cp_string]
    x3 = [s.split("%")[0] for s in x3]

    x1 = [float(s) for s in x1]
    x2 = [float(s) for s in x2]
    x3 = [float(s) for s in x3]
    
    X = np.zeros([len(cp_string), 3])
    
    X[:,0] = x1
    X[:,1] = x2
    X[:,2] = x3

    return X


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

        
def data_summary_plot(bat_dict, save_plots, path_save='', file_name='cycle_life_summary.html'):
    # Visualization of Data Summary
    bat_keys_arr = [i for i in bat_dict.keys()]

    fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in bat_keys_arr[:]:
        if i[:2]=='b3':
            name='b8'+i[2:]
        else:
            name=i
        fig.add_trace(go.Scatter(x=bat_dict[i]['summary']['cycle'][:], y=bat_dict[i]['summary']['QD'][:], name=name,
                                 marker = dict(color=color_map_rgb(len(bat_dict[i]['summary']['QD'][:])/2300))),
                      row=1, col=1)

    fig.update_yaxes(title_text="Discharge Capacity (Ah)", row=1, col=1)
    fig.update_xaxes(title_text="Cycle Number", row=1, col=1)
    fig.update_layout(title_text="Battery Cycle Life", autosize=False, width=1000, height=600,)

    fig.show()
    if save_plots:
        fig.write_html(path_save + file_name)
    
    return None
    
    
def single_cycle_plot(bat_dict, bat_key, cycle=1, save=0):
    fig, host = plt.subplots(figsize=(12,10))
    #fig.subplots_adjust(right=0.75)
    
    par1 = host.twinx()
    par2 = host.twinx()
    
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.13))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)
    
    p3, = par2.plot(bat_dict[bat_key]['cycles'][str(cycle)]['t'], 'b', linewidth=2.5, label='time')
    
    p1, = host.plot(np.arange(len(bat_dict[bat_key]['cycles'][str(cycle)]['t'])), 
                    bat_dict[bat_key]['cycles'][str(cycle)]['I'], 'r', linewidth=2.5, label='current')
    
    p2, = par1.plot(np.arange(len(bat_dict[bat_key]['cycles']['1']['t'])), 
                    bat_dict[bat_key]['cycles'][str(cycle)]['V'], 'g', linewidth=2.5, label='voltage')
    
    
    #host.set_xlim(0, 2)
    #host.set_ylim(0, 2)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(1, 65)
    cprot = bat_dict[bat_key]['charge_policy']
    # host.set_title("Single Cycle, Battery: " + bat_key + '  Charging Protocol: ' + cprot)
    host.set_title('Example Cycle Charging Protocol: ' + cprot, fontsize=28)
    host.set_xlabel("Index",  fontsize=25)
    host.set_ylabel("Current (C-Rate)",  fontsize=25)
    par1.set_ylabel("Voltage (V)",  fontsize=25)
    par2.set_ylabel("Time (min)",  fontsize=25)
    max_i = bat_dict[bat_key]['cycles']['1']['I'].max()
    min_i = bat_dict[bat_key]['cycles']['1']['I'].min()
    host.set_yticks(np.arange(np.int(min_i -1), np.int(max_i+1), step=1))
    
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)
    
    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines], loc=1, bbox_to_anchor=(0.85,1))
    # host.grid()
    
    if save: 
        plt.savefig('../../03Results/01Visualizations/single_cycle.pdf', bbox_inches='tight')
        
    plt.show()
    
    return None 


def plot_discharge_ridge(bat_dict, first_c = 1, last_c = 300, key = ['b1c1'], pos_cycle=0, 
                        x_text='Cycle', y_text='Voltage (V)', z_text='Qd (Ah)', title_text='Q(V) X', 
                        c_pos=dict(x=1.5, y=1.5, z=1.), interpolate_smoothing=1, save=0):
    
    points = 50
    freq = int(1000/points)
    diff = np.zeros([points, last_c-first_c])
    diff_smooth = np.zeros([points, last_c-first_c])
    test = extract_Qdlin(bat_dict, key, max_cycle=last_c+1).squeeze()[::freq,]

    if pos_cycle == 0:
        for i in np.arange(first_c, last_c):
            diff[:,i-first_c] = test[:, i]
            
        title_0 = "Discharge Waterfall"
    else:
        for i in np.arange(first_c, last_c):
            diff[:,i-first_c] = test[:, pos_cycle-1]-test[:, i]
        title_0 = "Discharge Ridge"

    y = np.linspace(3.5,2,points)
    x = np.linspace(first_c, last_c, last_c-first_c)
    
    if interpolate_smoothing: 
        #Do a liner regression for each voltage section and calculate a new Z. 
        for i,v in enumerate(y): 
            huber = HuberRegressor().fit(x.reshape(-1, 1), diff[i,:])
            diff_smooth[i, :] = huber.predict(x.reshape(-1, 1))
        fig = go.Figure(data=[go.Surface(y=y, z=diff_smooth[:])])
    else:
        fig = go.Figure(data=[go.Surface(y=y, z=diff[:])])
           
    fig.update_layout(
        title={
            'text': title_0 + ', Battery: ' + str(key[0]) +'   ' + title_text,
            'y':0.87,
            'x':0.1,
            'xanchor': 'left',
            'yanchor': 'top'})
    
    fig.update_layout(autosize=True,
                      width=1000, height=1000,
                      margin=dict(l=0, r=0, b=0, t=50))
    
    fig.update_layout(title_font_size=30)
    
    camera = dict(
        eye=c_pos
    )
    fig.update_layout(scene_camera=camera)
    
    fig.update_scenes(yaxis_title=dict(text=y_text, font_size=25))
    fig.update_scenes(xaxis_title=dict(text=x_text, font_size=25))
    fig.update_scenes(zaxis_title=dict(text=z_text, font_size=25))
    
    fig.update_layout(scene = dict(
                    xaxis = dict(
                        tickfont_size=15),
                    yaxis = dict(
                        tickfont_size=15),
                    zaxis = dict(
                        tickfont_size=15)))
    
    fig.update_traces(colorbar=dict(lenmode='fraction', 
                                    len=0.65, 
                                    thickness=23, 
                                    title='Qd (Ah)', 
                                    title_font_size=24,
                                    tickfont=dict(size=15)))

    if save:
        fig.write_image('../../03Results/01Visualizations/' + title_0 + str(key[0]) + title_text + '.png', 
                        engine='kaleido', 
                        scale=2)

    fig.show()

def multi_cycle_multivar_plot(bat_dict, bat_key, first_cycle=0, last_cycle=10, save=0):
    I = np.array([])
    Qc = np.array([])
    Qd = np.array([])
    Qdlin = np.array([])
    T = np.array([])
    Tdlin = np.array([]) 
    V = np.array([]) 
    dQdV = np.array([]) 
    t = np.array([])
    
    for i in np.arange(first_cycle,last_cycle):
        I = np.append(I, bat_dict[bat_key]['cycles'][str(i)]['I'])
        Qc = np.append(Qc, bat_dict[bat_key]['cycles'][str(i)]['Qc'])
        Qd = np.append(Qd, bat_dict[bat_key]['cycles'][str(i)]['Qd'])
        Qdlin = np.append(Qdlin, bat_dict[bat_key]['cycles'][str(i)]['Qdlin'])
        T = np.append(T, bat_dict[bat_key]['cycles'][str(i)]['T'])
        Tdlin = np.append(Tdlin, bat_dict[bat_key]['cycles'][str(i)]['Tdlin'])
        V = np.append(V, bat_dict[bat_key]['cycles'][str(i)]['V'])
        dQdV = np.append(dQdV, bat_dict[bat_key]['cycles'][str(i)]['dQdV'])
        if i==0:
            tnext = bat_dict[bat_key]['cycles'][str(i)]['t']
        else:
            tnext = bat_dict[bat_key]['cycles'][str(i)]['t'] + t[-1]
        t = np.append(t, tnext)

        
        
    # x = df_batdat.iloc[:length, 4]
    # u_savgol = savgol_filter(df_batdat.iloc[:length, 2], 301, 1)
    # peaks, _ = find_peaks(u_savgol, prominence=0.03)
    # peaks_rawvolt, _ = find_peaks(df_batdat.iloc[:length, 2], width=60, prominence=0.13, distance=300)
    # peaks = np.insert(peaks, 0, 0)
    # peaks_rawvolt = np.insert(peaks, 0, 0)

    
    fig = make_subplots(rows=8, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=t, y=I, name="current"), row=1, col=1)
    fig.update_yaxes(title_text="I (A)", row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=Qc, name="charge capacity"), row=2, col=1)
    fig.update_yaxes(title_text='Cap C (Ah)', row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=Qd, name="discharge capacity"), row=3, col=1)
    fig.update_yaxes(title_text='Cap D (Ah)', row=3, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=V, name="Voltage"), row=4, col=1)
    fig.update_yaxes(title_text='(V)', row=4, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=T, name="Temperature"), row=5, col=1)
    fig.update_yaxes(title_text='T (°C)', row=5, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=Qdlin, name='Qdlin ?'), row=6, col=1)
    fig.update_yaxes(title_text='Qdlin', row=6, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=Tdlin, name='Tdlin?'), row=7, col=1)
    fig.update_yaxes(title_text='Tdlin', row=7, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=dQdV, name='dQdV?'), row=8, col=1)
    fig.update_yaxes(title_text='dQdV', row=8, col=1)
    

    fig.update_layout(title_text="Battery Cycle Data Visualization Cell " + bat_key)

    fig.show()
    
    if save==1:
        fig.write_html(
            '../../03Results/01Visualizations/' + bat_key + 'cycle_data' \
            + str(first_cycle) + '-' + str(last_cycle) + '.html')
    
    return None


def V_QDlin_plot(bat_dict, bat_key_vis, first_cycle=3, last_cycle=110):
    # Visualization based on the Code of Peter Attia (https://github.com/petermattia/predicting-battery-lifetime)
    # last_cycle =  len(bat_dict[bat_key]['cycles'][str(i)]['Qdlin'])
    colors = cm.get_cmap('viridis_r')(np.linspace(0, 1, last_cycle-first_cycle))

    Vdlin = np.linspace(3.6,2,1000)
    Qdlin = np.expand_dims(np.array(bat_dict[bat_key_vis]['cycles'][str(first_cycle)]['Qdlin']).T, axis =1)
    
    for i in np.arange(first_cycle+1,last_cycle):
        Qdlin = np.concatenate(
            (Qdlin, np.expand_dims(np.array(bat_dict[bat_key_vis]['cycles'][str(i)]['Qdlin']).T, axis = 1)), axis=1)
    
    for k in np.arange(last_cycle-first_cycle):
        plt.plot(Qdlin[:,k],Vdlin,color=colors[k])
        
    plt.xlabel('Discharge capacity (Ah)', fontsize=17)
    plt.ylabel('Voltage (V)', fontsize=17)
    plt.xlim((0,1.1))
    plt.ylim((2,3.6))
    plt.arrow(0.95, 0.6, -0.2, 0, transform=plt.gca().transAxes,zorder=3,head_width=0.02,color='k')
    plt.text(0.5, 0.49, 'Increasing\ncycle number', horizontalalignment='center',
         verticalalignment='center',transform=plt.gca().transAxes)
    
    plt.show()
    
    return None

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #Source: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def V_QDlin_subplot(bat_dict, bat_key_vis, ax, first_cycle=3, last_cycle=110):
    # Visualization based on the Code of Peter Attia (https://github.com/petermattia/predicting-battery-lifetime)
    # last_cycle =  len(bat_dict[bat_key]['cycles'][str(i)]['Qdlin'])
    cmap = truncate_colormap(cm.get_cmap('plasma'), 0.0, 0.8)(np.linspace(0, 1, last_cycle-first_cycle+1))
    #colors = cm.get_cmap('viridis_r')(np.linspace(0, 1, last_cycle-first_cycle+1))
    #colors = cm.get_cmap('winter')(np.linspace(0, 1, last_cycle-first_cycle+1))
    Vdlin = np.linspace(3.6,2,1000)
    Qdlin = np.expand_dims(np.array(bat_dict[bat_key_vis]['cycles'][str(first_cycle)]['Qdlin']).T, axis =1)
    
    for i in np.arange(first_cycle+1,last_cycle+1):
        Qdlin = np.concatenate(
            (Qdlin, np.expand_dims(np.array(bat_dict[bat_key_vis]['cycles'][str(i)]['Qdlin']).T, axis = 1)), axis=1)
    
    for k in range(last_cycle+1-first_cycle):
        ax.plot(Qdlin[:,k],Vdlin,color=cmap[k])
       
        
    #ax.set_xlabel('Discharge capacity (Ah)')
    #ax.set_ylabel('Voltage (V)')
    ax.set_xlim((0,1.1))
    ax.set_ylim((2,3.6))
    
    
    return None


def R2_pred_pls_pcr(rows, cols, X_train__dict, y_train, X_test_dict, y_test, feature_keys, f_nb, method='PLS'):
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5)) #, constrained_layout=True)
    for i, (k, l) in enumerate(itertools.product(range(rows), range(cols))):   
        if method == 'PLS':
            pls = PLSRegression(n_components=i+1, scale=False)
            pls.fit(X_train__dict[feature_keys[f_nb]], y_train)
            y_pred = pls.predict(X_test_dict[feature_keys[f_nb]])
        elif method == 'PCR':
            pcr = make_pipeline(StandardScaler(), PCA(n_components=i+1), LinearRegression())
            pcr.fit(X_train__dict[feature_keys[f_nb]], y_train)
            y_pred = pcr.predict(X_test_dict[feature_keys[f_nb]])
        else:
            raise NameError('Method \"' + method + '\" is not implemeted!')
            

        sort_inds = y_test.argsort()
        sorted_y_test = y_test[sort_inds[::-1]]
        sorted_y_pred = y_pred[sort_inds[::-1]]

        axs[k, l].scatter(10**sorted_y_test, 10**sorted_y_pred)
        axs[k, l].plot(np.arange(2000), 'k')
        axs[k, l].set(xlabel='Cycle life test', ylabel='Cycle life pred', title=method +' with ' + str(i+1) + ' components')
        axs[k, l].set_xlim(0,2000)
        axs[k, l].set_ylim(0,2000)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(np.power(10, y_test), np.power(10, y_pred), squared=False)


        axs[k, l].text(50, 1600, '$R^2$ ={:.2f} \n RMSE = {:.1f}'.format(r2, rmse), fontsize=15,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    fig.suptitle(method + ' ' + feature_keys[f_nb])

    return None


def PLS_projected_prediction_plot(pls, X_train_dict, y_train, X_test_dict, y_test, feature_key, save=1):
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].scatter(pls.transform(X_test_dict[feature_key])[:, 0], y_test, alpha=.3, label='ground truth')
    ax[0].scatter(pls.transform(X_test_dict[feature_key])[:, 0], pls.predict(X_test_dict[feature_key]), alpha=.3,
                    label='predictions')
    ax[0].set_xlabel('Data Projected on First PLS Component', fontsize=14)
    ax[0].set_ylabel('Log10(Cycle Life)', fontsize=14)
    ax[0].set_title('PLS ' + feature_key + ' Test Dataset (Batch3)', fontsize=19)
    ax[0].legend()
    ax[0].grid()

    ax[1].scatter(pls.transform(X_train_dict[feature_key])[:, 0], y_train, alpha=.3, label='ground truth')
    ax[1].scatter(pls.transform(X_train_dict[feature_key])[:, 0], pls.predict(X_train_dict[feature_key]), alpha=.3,
                    label='predictions')
    
    ax[1].set_xlabel('Data Projected on First PLS Component', fontsize=14)
    #ax[1].set_ylabel('Log10(Cycle Life)', fontsize=14)
    ax[1].set_title('PLS ' + feature_key + ' Train Dataset (Batch1&2)', fontsize=19)
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    if save: 
        plt.savefig('../../03Results/01Visualizations/PLS_projection.png', bbox_inches='tight', dpi=300)
    plt.show()

    return None 

def PCR_projected_prediction_plot(evecs_pca, pcr_coef_, pcr_const, X_train__dict, y_train, X_test__dict, y_test, feature_key):
    X_train_proj = np.dot(X_train__dict[feature_key], evecs_pca[:,0])
    y_train_pred = np.dot(X_train__dict[feature_key], pcr_coef_)+pcr_const
    X_test_proj = np.dot(X_test__dict[feature_key], evecs_pca[:,0])
    y_test_pred = np.dot(X_test__dict[feature_key], pcr_coef_)+pcr_const
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].scatter(X_test_proj, y_test, alpha=.3, label='ground truth')
    ax[0].scatter(X_test_proj, y_test_pred, alpha=.3,
                    label='predictions')
    ax[0].set(xlabel='Projected data onto first PLS component',
                ylabel='y', title='PCR ' + feature_key + ' Test Dataset')
    ax[0].legend()

    ax[1].scatter(X_train_proj, y_train, alpha=.3, label='ground truth')
    ax[1].scatter(X_train_proj, y_train_pred, alpha=.3,
                    label='predictions')
    ax[1].set(xlabel='Projected data onto first PLS component',
                title='PCR ' + feature_key + ' Train Dataset')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return None 

def PLS_coefficient_plot(X_train_dict, feature_keys, f_nb, y_train, max_comp=5, save=0):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    Vdlin = np.linspace(3.6,2,1000)

    for i in range(max_comp):
        pls = PLSRegression(n_components=i+1, tol=1e-7, scale=False)
        pls.fit(X_train_dict[feature_keys[f_nb]], y_train)
        fig.add_trace(go.Scatter(x=Vdlin, y=pls.coef_.reshape(-1), name=str(i+1)+' comp'), row=1, col=1)

        
    fig.update_yaxes(title_text='PLS Regression Coefficients', row=1, col=1)
    fig.update_xaxes(title_text='Voltage (V)', row=1, col=1)
    fig.update_layout(title_text='PLS ' + feature_keys[f_nb])

    
    fig.update_layout(
        title={
            'text': 'PLS Regression Coefficients',
            'y': 1,
            'x': 0.1,
            'xanchor': 'left',
            'yanchor': 'top' })

    fig.update_layout(autosize=False,
                      width=1400, height=700,
                      margin=dict(l=0, r=0, b=0, t=50))

    fig.update_layout(font=dict(size=18))    
    fig.show()
    
    if save==1:
        fig.write_html('../../03Results/01Visualizations/PLScomponentscoeff' + str(max_comp) + '.html')
        fig.write_image('../../03Results/01Visualizations/PLScomponentscoeff' + str(max_comp) + '.png', engine='kaleido')
    return None 

def PCR_coefficient_plot(X_train__dict, feature_keys, f_nb, y_train, max_comp=5, save=0):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    Vdlin = np.linspace(3.6,2,1000)

    for i in range(max_comp):
        X_pca, evecs_pca, pcr_coef_, pcr_const_ = src.pcr_custom(X_train__dict[feature_keys[f_nb]], y_train, dims=i+1)
        fig.add_trace(go.Scatter(x=Vdlin, y=pcr_coef_, name=str(i+1)+' comp'), row=1, col=1)

    fig.update_yaxes(title_text='Abs. Regression Coefficients', row=1, col=1)
    fig.update_xaxes(title_text='Voltage (V)', row=1, col=1)
    fig.update_layout(title_text='PCR ' + feature_keys[f_nb])

    fig.show()
    
    if save==1 :
        fig.write_html('../../03Results/01Visualizations/PCRcomponentscoeff.html')

    return None 

def plot_cp(c_train, c_test1, c_test2, title_string='Charging Protocols 3D Visualization', convex_hull=False, save=0): 
    
    colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]
    
    trace = go.Scatter3d(
                    x = c_train[:,0], 
                    y = c_train[:,1], 
                    z = c_train[:,2], 
                    mode = 'markers', 
                    marker = dict(size=5, color = colors[0]), 
                    name='Batch 1')
    
    trace1 = go.Scatter3d(
                    x = c_test1[:,0], 
                    y = c_test1[:,1], 
                    z = c_test1[:,2], 
                    mode = 'markers', 
                    marker = dict(size=5, color = colors[1]), 
                    name='Batch 2')
    
    trace2 = go.Scatter3d(
                    x = c_test2[:,0], 
                    y = c_test2[:,1], 
                    z = c_test2[:,2], 
                    mode = 'markers', 
                    marker = dict(size=5, color = colors[2]), 
                    name='Batch 8')

    layout = go.Layout(title=title_string, height=800, width=1000, legend={'traceorder':'normal'})
    
    c_all =  np.append(c_train, np.append(c_test1, c_test2,axis=0), axis=0)
    
    if convex_hull:
        xc = c_all[ConvexHull(c_all).vertices]
        trace_cvx_h = go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color= 'blue', 
                        opacity=.15,
                        alphahull=0)
        fig = go.Figure(data = [trace, trace1, trace2, trace_cvx_h], layout=layout)    
    else:
        fig = go.Figure(data = [trace, trace1, trace2], layout=layout)
        
    fig.update_scenes(yaxis_title=dict(text='C-Rate Start'))
    fig.update_scenes(yaxis_title_font=dict(size=16))
                      
    fig.update_scenes(xaxis_title=dict(text='C-Rate End'))
    fig.update_scenes(xaxis_title_font=dict(size=16))
                      
    fig.update_scenes(zaxis_title=dict(text='SOC (%) Change C-Rate'))
    fig.update_scenes(zaxis_title_font=dict(size=16))
        
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7,
        font=dict(
            size=16,
            color="black")
        )
    )
    
    config = {
        'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 500 # Multiply title/legend/axis/canvas sizes by this factor
      }
    }
    fig.show(config=config)
    if save:
        fig.write_image('../../03Results/01Visualizations/cp_batch123.png', engine='kaleido')
   
    return None
    

def plot_deltaQ(bat_dict, bat_keys, high_cycle_index, low_cycle_index, save_plots):
    
    Qdlin = extract_Qdlin(bat_dict, bat_keys, high_cycle_index)
    DeltaQ_100_minus_10 = Qdlin[:, :, high_cycle_index-1] - Qdlin[:, :, low_cycle_index-1]

    y = extract_cyclelife(bat_dict, bat_keys)
    y_norm = y/y.max()
    Vdlin = np.linspace(3.5,2,1000)
    cmap = truncate_colormap(cm.get_cmap('plasma'), 0.05, 0.85)
    
    fig, ax = plt.subplots()

    for k in np.arange(len(DeltaQ_100_minus_10)):
        ax.plot(Vdlin, DeltaQ_100_minus_10[k, :], color=cmap(y_norm[k]), linewidth=1.3)

    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=2250, clip=False), cmap=cmap), ax=ax)
    cb.set_label('Cycle Life', labelpad=10,fontsize=25)
    plt.xlabel('Voltage (V)', fontsize=25)
    plt.ylabel(r'$\Delta Q_{100-10}$ (Ah)', fontsize=25)
    plt.xlim([2.0,3.5])
    plt.title('Discharge Capacity Difference Cycle 100-10', fontsize=28)
    #plt.grid()
    fig.tight_layout()
    if save_plots: 
        plt.savefig('../../03Results/01Visualizations/DQ100_10Batch123_midterm.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    return None


def check_uniqueness_cprotocol(bat_dict, bat_keys, keys_set1, keys_set2, keys_set3=np.empty(0)):
    # extract charge policy information
    charge_policies = extract_charge_policy(bat_dict, bat_keys)
    keys_charge = list(zip(bat_keys, charge_policies))

    # create empty dict 
    chargepol_batkey_dict = defaultdict(list)
    batkey_chargepol_dict = defaultdict(list)

    for i, j in zip(charge_policies, bat_keys):
        chargepol_batkey_dict[i].append(j)

    for i, j in zip(bat_keys, charge_policies):
        batkey_chargepol_dict[i].append(j)

    # create list of charge policies in the train and in the test dataset 
    train_charge_policies = extract_charge_policy(bat_dict, keys_set1)
    test1_charge_policies = extract_charge_policy(bat_dict, keys_set2)
    if keys_set3.size != 0:
        test2_charge_policies = extract_charge_policy(bat_dict, keys_set3)

    # Let's check which charge policies are violiting the grouping constraint for cross validation
    pol_nonun_tr_t1 = (set(train_charge_policies) & set(test1_charge_policies))
    if keys_set3.size !=0:
        pol_nonun_tr_t2 = (set(train_charge_policies) & set(test2_charge_policies))
        pol_nonun_t1_t2 = (set(test1_charge_policies) & set(test2_charge_policies))

    print("Approaches in set 1 as well as set 2: " + str(pol_nonun_tr_t1))
    if keys_set3.size !=0:
        print("Approaches in set 1 as well as set 3: " + str(pol_nonun_tr_t2))
        print("Approaches in set 2 as well as set 3: " + str(pol_nonun_t1_t2))
    
    return None