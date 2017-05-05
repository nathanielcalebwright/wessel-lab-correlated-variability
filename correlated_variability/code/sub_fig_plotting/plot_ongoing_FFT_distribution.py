import numpy
import matplotlib.ticker as plticker

from analysis.get_low_freq_FFT_all_cells import \
get_low_freq_FFT_all_cells as get_FFT

import matplotlib as mpl


def plot_ongoing_FFT_distribution(ax, fontproperties,
        stim_types = ['extended_stim', 'flash'],
        windows = [9000.0, 2000.0, 1000.0], 
        gaps = [0.0, 200.0, 200.0],
        padding = 0.0,
        freq_range_for_FFT = (1.0, 5.0),
        replace_spikes = True, remove_sine_waves = True,
        get_new_FFT_results_all_cells = False,
        get_new_processed_traces = False,
        master_folder_path = 'E:\\correlated_variability'):

    """
    For all recorded neurons calculate the summed low-frequency Fourier 
    coefficients for pre-stimulus activity, and plot the histrogram of
    population result.
    
    Parameters
    ----------
    ax: matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    stim_types : python list of strings
        include all cells corresponding to these stimulus types (e.g., 
        ['extended_stim', 'flash'])
    windows: list of floats
        widths of ongoing, transient, and steady-state windows (ms).  The FFT 
        will be calulated for all windows, but only the result for the ongoing
        window is eventually used.
    gaps: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms).
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace
    freq_range_for_FFT: tuple of floats
        range of frequencies over which to sum Fourier coefficients (e.g., 
        (1.0, 5.0), as in Sachidhanandam, et al. 2013)
    get_new_FFT_results_all_cells: bool
        if True, calculate ongoing FFT for unfiltered sub-traces.  otherwise, 
        use saved results (numpy dictionaries) if they exist, to speed things 
        up.
    get_new_processed_traces: bool
        if True, start with unprocessed data (actually, downampled to 1 kHz
        from 10 kHz).  otherwise, use saved subtraces (numpy arrays) if they
        exist, to speed things up.
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    none
    """

    fonts = fontproperties
    size = fonts.get_size()
    mpl.rcParams['mathtext.default'] = 'regular'
    
    FFT_dict_all_cells = get_FFT(stim_types, windows, gaps, padding,
        freq_range_for_FFT, replace_spikes, remove_sine_waves,
        get_new_FFT_results_all_cells, get_new_processed_traces,
        master_folder_path)
                            
    FFTs_all_cells = []
    for cell in FFT_dict_all_cells:
        FFTs_all_cells.append(numpy.mean(FFT_dict_all_cells[cell]['ongoing']))

    ### plot the histogram ###

    hist1 = ax.hist(FFTs_all_cells, bins = 15, color = 'k', alpha = 0.4)

    ### format the histogram ###    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.get_xaxis().tick_bottom()

    ax.tick_params(axis = 'x', length = 3.0, width = 1.0, direction = 'outward')
    loc = plticker.MultipleLocator(base = 200.0)
    ax.xaxis.set_major_locator(loc)
    x_labels = ax.get_xticks()
    ax.set_xticklabels(x_labels, fontsize = size)
    ax.set_xlabel(r'$\langle FFT_\delta\rangle$ (mV)', fontsize = size)   

    ### report number of cells and preparations ###
    experiments = []
    for cell_name in FFT_dict_all_cells:
        expt_date = cell_name.split('_')[0]
        if expt_date not in experiments:
            experiments.append(expt_date)
    print '    %s cells;'%str(len(FFT_dict_all_cells.keys())), \
          '%s turtles'%str(len(experiments))
    
