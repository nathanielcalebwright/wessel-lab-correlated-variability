'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

import numpy
import os
from signal_processing.get_processed_subtraces_for_cell import \
get_processed_subtraces_for_cell as get_processed_subtraces

def plot_example_high_freq_residuals_for_cell_pair(ax, fontproperties,
        cells_to_plot = ['070314_c2', '070314_c3'], 
        trial_nums_to_plot = [0, 1, 2, 29],
        windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0], 
        padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'low', 
        replace_spikes = True, remove_sine_waves = True,
        get_new_processed_traces = False,
        master_folder_path = 'E:\\correlated_variability'):

    """
    For an example pair of simultaneously-recorded neurons, plot (in low
    responses to multiple presentations of a visual stimulus,
    Process the traces, if not already done (replace spikes, remove sine waves,
    and filter (according to crit_freq and filt_kind)).  Stack the trials
    vertically, and label trial numbers.  
    Note: plot formatting is based on the default filter settings, and for the 
    pair and trials displayed in the manuscript figure.
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    cells_to_plot : python list of strings
        list of cells to plot (e.g., ['070314_c2', '070314_c3'])
    trial_to_plot : python list of ints
        list of trials to plot (e.g., [0, 1, 2, 29])
    windows : list of floats
        widths of ongoing, transient, and steady-state windows (ms)
    gaps : list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms)
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace (only nonzero when doing wavelet filtering)
    crit_freq: float, or tuple of floats
        critical frequency for broad-band filtering of traces (e.g., 100.0)
    filt_kind: string
        type of filter to be used for traces (e.g., 'low' for lowpass)   
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
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
    
    colors = ['k', 'r'] #trace colors for top and bottom cells in plot
    
    #pull out sections of interest for each stim presentation    
    for j, cell_name in enumerate(cells_to_plot):
        traces_path = master_folder_path + '\\processed_sub_traces\\'
        traces_path += cell_name + '_'       
        traces_path += str(windows) + '_ms_windows_'
        traces_path += str(gaps) + '_ms_gaps_'
        traces_path += str(padding) + '_ms_padding_'
        traces_path += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
        if replace_spikes:
            traces_path += '_spikes_removed'
        if remove_sine_waves:
            traces_path += '_sine_removed'
        traces_path += '.npy'
        
        #see if these processed subtraces already exist
        if os.path.isfile(traces_path) and get_new_processed_traces == False:
            subtraces_for_cell = numpy.load(traces_path)
        #if not, get from raw data
        else:
            subtraces_for_cell = get_processed_subtraces(cell_name, windows,
                gaps, padding, crit_freq, filt_kind, replace_spikes,
                remove_sine_waves, master_folder_path)
            
            numpy.save(traces_path, subtraces_for_cell)
        
        ### plot residuals for trials of interest ###
        resids_for_cell = subtraces_for_cell - numpy.mean(subtraces_for_cell,
            axis = 0)
        offset = 0
        trial_num = 0
        trial_nums = [0, 1, 2, 29]
        offsets = [-2, -7, -12, -17]
        color = colors[j]
        for k, trial_num in enumerate(trial_nums):
            offset = offsets[k]
            resid = resids_for_cell[k]
            ax.plot(resid + offset, color = color, lw = 0.5)
                 
                            
    ### turn off axes and tick marks ###    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    ### add scale bar ###
    V_bar_xs = [400, 400]
    V_bar_ys = [0.4, 2.4]
    
    t_bar_xs = [400, 900]
    t_bar_ys = [0.4, 0.4]
    
    ax.plot(V_bar_xs, V_bar_ys, 'k', lw = 1.5)
    ax.plot(t_bar_xs, t_bar_ys, 'k', lw = 1.5)
    
    ax.text(t_bar_xs[0] - 30, 1, '2 mV', fontsize = size,
        horizontalalignment = 'right')

    ax.text(numpy.mean(t_bar_xs), t_bar_ys[0] - 1.0, '500 ms', fontsize = size,
        horizontalalignment = 'center')
    
    ### add trial number labels ###    
    for n, trial_num in enumerate(trial_nums):
        label = 'trial ' + str(trial_num + 1)
        y_pos = offsets[n] - 1.5
        ax.text(470, y_pos, label, fontsize = size,
                horizontalalignment = 'center')
        
    dot_labels = ['.', '.', '.']
    
    offset = -14.5   
    for label in dot_labels:
        ax.text(470, offset, label, fontsize = size, 
            horizontalalignment = 'center')
        offset += -1
    
    