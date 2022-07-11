import numpy as np
import matplotlib.colors as mcolors

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