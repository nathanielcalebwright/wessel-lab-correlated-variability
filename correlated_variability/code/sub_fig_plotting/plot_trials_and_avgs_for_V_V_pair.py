'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

import pylab
import numpy
import os
from signal_processing.get_processed_subtraces_for_cell import \
get_processed_subtraces_for_cell as get_processed_subtraces

def plot_trials_and_avgs_for_V_V_pair(ax, fontproperties,
        cells_to_plot = ['070314_c2', '070314_c3'],
        windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0], 
        padding = 0.0, crit_freq = 100, filt_kind = 'low', offset = -35, 
        replace_spikes = True, remove_sine_waves = True,
        get_new_processed_traces = False,
        master_folder_path = 'E:\\correlated_variability'):

    """
    For an example pair of simultaneously-recorded neurons, plot (in low
    opacity) responses to multiple presentations of a visual stimulus,
    as well as the across-trial averages (in high opacity).  Process the 
    traces, if not already done (i.e., downsample to 1 kHz, remove spikes,
    and filter (100 Hz lowpass Butterworth)).  For each cell, align
    the traces to 5th percentile of pre-stimulus Vm (for clarity).
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    cells_to_plot : list of strings
        list of cells to plot (e.g., ['070314_c2', '070314_c3'])
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
    offset : float or int
        imposed vertical distance (mV) between traces for one cell and
        traces for the other    
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

    #background colors for each window of activity
    y = (1, 1, 0)
    g = (0, 1, 0)
    b = (0.2, 0.6, 1)
    
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

        #manually align the traces
        subtraces_for_plot = []
        for subtrace in subtraces_for_cell:
            new_subtrace = subtrace - numpy.percentile(subtrace[:int(padding + windows[0])], 5)
            subtraces_for_plot.append(new_subtrace)
        subtraces_for_plot = pylab.array(subtraces_for_plot)
        
        
        #plot trials and averages
        color = colors[j]
        if j == 1:
            offset_for_plot = offset
        else:
            offset_for_plot = 0
        for sub_trace in subtraces_for_plot:
            ax.plot(sub_trace + offset_for_plot, color = color, alpha = 0.2, 
                lw = 0.65)
        
        mean_response = numpy.mean(pylab.array(subtraces_for_plot), 
            axis = 0)
        ax.plot(mean_response + offset_for_plot, color = color, lw = 1.5)            
                            
    #configure the axis
    ax.set_xlim(padding, (sum(windows) + sum(gaps) + 2*padding))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #indicate epoch windows with color
    ax.axvspan(padding, padding + windows[0], ymin = 0.1, ymax = 1, 
        facecolor = y, ec = 'none', alpha = 0.3)
    
    ax.axvspan(padding + windows[0] + sum(gaps[:2]), 
        padding + sum(windows[:2]) + sum(gaps[:2]), 
        ymin = 0.1, ymax = 1, facecolor = b, ec = 'none', alpha = 0.3)
    
    ax.axvspan(padding + sum(gaps) + sum(windows[:2]), 
        padding + sum(gaps) + sum(windows), 
        ymin = 0.1, ymax = 1, facecolor = g, ec = 'none', alpha = 0.3)
    
    #add scale bars
    V_bar_xs = [padding + 750, padding + 750]
    V_bar_ys = [20, 30]
    V_text = '10 mV'
    
    t_bar_xs = [padding + 750, padding + 1750]
    t_bar_ys = [20, 20]
    V_text_y = t_bar_ys[0] + 2
    t_text_y = t_bar_ys[0] - 6

    
    ax.plot(V_bar_xs, V_bar_ys, 'k', lw = 1.5)
    ax.plot(t_bar_xs, t_bar_ys, 'k', lw = 1.5)
    
    ax.text(V_bar_xs[0] - 50, V_text_y, V_text, fontsize = size,
        horizontalalignment = 'right')

    ax.text(numpy.mean(t_bar_xs), t_text_y, '1 s', fontsize = size,
        horizontalalignment = 'center')
    
    