'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

from signal_processing.get_processed_subtraces_for_cell import \
get_processed_subtraces_for_cell as get_processed_subtraces

from matplotlib.patches import ConnectionPatch
import os
import numpy

def plot_example_trace_with_zoom_inset(ax1, ax2, fontproperties,
        cell_name = '070314_c2', trial_num = 2,
        windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
        padding = 0.0, crit_freq = 100.0, filt_kind = 'low',
        replace_spikes = True, remove_sine_waves = True,
        master_folder_path = 'E:\\correlated_variability'):

    """
    For an example neuron, plot a single-trial visual response
    with a zoomed view of high-conductance activity (demonstrating the
    "nested" higher-frequency activity within the broader depolarization).
    Process the all traces for this cell, if not already done 
    (i.e., remove spikes, filter, remove sine waves).
    
    Parameters
    ----------
    ax1 : matplotlib axis
        for full response window
    ax2 : matplotlib axis
        for zoomed view
    fontproperties: FontProperties object
        dictates size and font of text
    cell_name : string
        name of example cell to plot (e.g., '070314_c2')
    trial_num: int
        index of trial to plot (where 0 is first trial)
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
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    none
    """
    
    fonts = fontproperties
    size = fonts.get_size()

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
    if os.path.isfile(traces_path):
        subtraces_for_cell = numpy.load(traces_path)
    #if not, get from raw data
    else:
        subtraces_for_cell = get_processed_subtraces(cell_name, windows,
                gaps, padding, crit_freq, filt_kind, replace_spikes,
                remove_sine_waves, master_folder_path)
        numpy.save(traces_path, subtraces_for_cell)
    
    sub_trace = subtraces_for_cell[int(trial_num)]

    ax1.plot(sub_trace, color = 'r', lw = 1.0)
    
    #configure the larger axis
    ax1.set_xlim(0, (sum(windows) + sum(gaps)))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    
    #draw a box around section of interest on larger axis
    box_xmin = 1650
    box_xmax = box_xmin + 500
    
    ax1.hlines(min(sub_trace[box_xmin:box_xmax]) - 1, box_xmin, box_xmax, 
        alpha = 0.5)
    ax1.hlines(max(sub_trace[box_xmin:box_xmax]) + 1, box_xmin, box_xmax, 
        alpha = 0.5)
    ax1.vlines(box_xmin, min(sub_trace[box_xmin:box_xmax]) - 1, 
        max(sub_trace[box_xmin:box_xmax]) + 1, alpha = 0.5)
    ax1.vlines(box_xmax, min(sub_trace[box_xmin:box_xmax]) - 1, 
        max(sub_trace[box_xmin:box_xmax]) + 1, alpha = 0.5)
    
    # add scale bars
    y_val = max(sub_trace) - 2.5
    V_bar_xs = [box_xmax + 450, box_xmax + 450]
    V_bar_ys = [y_val, y_val + 5]
    
    t_bar_xs = [box_xmax + 450, box_xmax + 950]
    t_bar_ys = [y_val, y_val]
    
    ax1.plot(V_bar_xs, V_bar_ys, 'k', lw = 1.5)
    ax1.plot(t_bar_xs, t_bar_ys, 'k', lw = 1.5)
    
    ax1.text(numpy.mean(t_bar_xs), y_val - 2.5, '500 ms', fontsize = size,
        horizontalalignment = 'center')

    ax1.text(t_bar_xs[0] + 50, numpy.mean(V_bar_ys), '5 mV', fontsize = size,
        horizontalalignment = 'left', verticalalignment = 'center')
            
    ### draw zoomed trace on other axis ###
    zoom_trace = sub_trace[box_xmin:box_xmax]
    ax2.plot(zoom_trace, 'r')
    ax2.set_xlim(-5, box_xmax - box_xmin + 5)
    
    ### draw box around zoomed plot ###    
    ax2.hlines(min(sub_trace[box_xmin:box_xmax]) - 1, 0, 
        box_xmax - box_xmin, alpha = 0.5)
    ax2.hlines(max(sub_trace[box_xmin:box_xmax]) + 1, 0,
        box_xmax - box_xmin, alpha = 0.5)
    ax2.vlines(0, min(sub_trace[box_xmin:box_xmax]) - 1, 
        max(sub_trace[box_xmin:box_xmax]) + 1,
        alpha = 0.5)
    ax2.vlines(box_xmax - box_xmin, 
        min(sub_trace[box_xmin:box_xmax]) - 1, 
        max(sub_trace[box_xmin:box_xmax]) + 1,
        alpha = 0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    #connect left corners of full window to right corners of zoomed
    #top corners
    xyA = (box_xmin, max(sub_trace[box_xmin:box_xmax]) + 1)
    xyB = (box_xmax - box_xmin, max(sub_trace[box_xmin:box_xmax]) + 1)
    coordsA = 'data'
    coordsB = 'data'
    con1 = ConnectionPatch(xyA, xyB, coordsA, coordsB, axesA=ax1, 
        axesB=ax2, color = 'k', alpha = 0.5)
    ax1.add_artist(con1)

    #bottom corners
    xyA = (box_xmin, min(sub_trace[box_xmin:box_xmax]) - 1)
    xyB = (box_xmax - box_xmin, min(sub_trace[box_xmin:box_xmax]) - 1)
    coordsA = 'data'
    coordsB = 'data'
    con2 = ConnectionPatch(xyA, xyB, coordsA, coordsB, axesA=ax1, 
        axesB=ax2, alpha = 0.5)    
    ax1.add_artist(con2)
