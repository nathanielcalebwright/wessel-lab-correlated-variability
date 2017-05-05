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
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib as mpl

from sub_fig_plotting.plot_example_traces_with_low_freq_FFT import \
plot_example_traces_with_low_freq_FFT as plot_traces

from sub_fig_plotting.plot_ongoing_FFT_distribution import \
plot_ongoing_FFT_distribution as plot_FFT_dist

from sub_fig_plotting.plot_stim_trace import plot_stim_trace

from sub_fig_plotting.plot_CC_vs_ongoing_FFT import \
plot_CC_vs_ongoing_FFT


def make_Fig4(example_cells_to_plot = ['030415_c1', '070314_c5'],
        stim_types = ['extended_stim', 'flash'],
        windows_for_FFT = [8000.0, 1000.0, 1000.0], 
        windows_for_CC = [1000.0, 1000.0, 1000.0], 
        gaps_for_FFT = [1000.0, 200.0, 200.0],
        gaps_for_CC = [0.0, 200.0, 200.0],
        padding_for_traces = 0.0,
        crit_freq_for_traces = 100.0, filt_kind_for_traces = 'low',
        crit_freq_for_CC = (20.0, 100.0), filt_kind_for_CC = 'band',
        freq_range_for_FFT = (1.0, 5.0),
        replace_spikes = True, remove_sine_waves = True,
        master_folder_path = 'F:\\correlated_variability_',
        savefig = False):

    '''
    Generate Figure 4 from the manuscript.


    Parameters
    ----------
    cells_to_plot : python list of strings
        list of cells to plot in panels A and B (e.g., ['070314_c2', 
        '070314_c3'])
    windows_for_FFT : list of floats
        widths of ongoing, transient, and steady-state windows (ms), for
        calculating ongoing FFT, and for plotting in A and B
    windows_for_CC : list of floats
        widths of ongoing, transient, and steady-state windows (ms), for
        calculating CC
    gaps : list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms), for all calculations and plots
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace, for all calculations and plots
    crit_freq_for_plot: float, or tuple of floats
        critical frequency for broad-band filtering of traces (e.g., 100.0),
        for plots in A and B
    filt_kind_for_plot: string
        type of filter to be used for traces (e.g., 'low' for lowpass), for
        plots in A and B
    crit_freq_for_CC: float, or tuple of floats
        critical frequency for broad-band filtering of traces (e.g., 100.0),
        for calculating CC
    filt_kind_for_CC: string
        type of filter to be used for traces (e.g., 'low' for lowpass), for
        calculating CC
    freq_range_for_FFT: tuple of floats
        range of frequencies (Hz) to use for summing low-freq ongoing FFT
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
    savefig: Bool
        if True, save the figure to the specified path
        
    Returns
    -------
    none
    '''

    
    width_fig = 5.0
    padding = 0.5 #space (in inches) b/w subfigs and edges of figure
    interpadding = 0.4 #reference unit for padding b/w subfigs
    
    fonts = FontProperties(); fonts.set_weight('bold'); fonts.set_size(6)
    fontm = FontProperties(); fontm.set_weight('bold'); fontm.set_size(12)
    fontl = FontProperties(); fontl.set_weight('bold'); fontl.set_size(12)
    mpl.rcParams['mathtext.default'] = 'regular'

    
    
    #a:multiple trials, V1-V2 (gamma band)
    #a2:stim trace for c
    #b:V-V CC trajectories (movies)
    #c:V-V CC trajectories (flashes)
    #d:trans CC vs. ong CC scatter
    #e:steady-state CC vs. trans CC scatter
    #f:steady-state CC vs. ong CC scatter
    #g:percent variance explained by each PC
    #h:percent variance in each epoch eplained by PC1
    
    #widths and heights of all subfigures IN INCHES
    side_square_plot = (1./4.)*(width_fig - 2*padding - interpadding)
    width_scatter = side_square_plot
    height_scatter = side_square_plot
    width_hist = side_square_plot
    height_hist = side_square_plot - 0.25*interpadding
    width_traces = width_fig - 2*padding - 2.0*interpadding - side_square_plot
    height_stim = 0.1
    height_fig = 4*side_square_plot + 2*padding + 3*interpadding
    
    height_traces = 0.5*(height_fig - 2*padding - 0.5*interpadding - 2*height_stim)
        
    
    #x and y positions of all subfigures and labels IN INCHES
    x_a = padding
    y_a = height_fig - padding - height_traces
    x_a2 = x_a
    y_a2 = y_a - height_stim
    x_b = x_a
    y_b = padding + height_stim
    x_b2 = x_b
    y_b2 = padding
    x_c = width_fig - padding - width_hist
    y_c = height_fig - padding - height_hist
    x_d = x_c
    y_d = y_c - 1.25*interpadding - height_scatter
    x_e = x_c
    y_e = y_d - interpadding - height_scatter
    x_f = x_c
    y_f = padding

    
    x_label_a = x_a - 0.3
    y_label_a = y_a + height_traces - 0.5*interpadding
    x_label_b = x_label_a
    y_label_b = y_e + height_hist
    x_label_c = x_c - 0.55
    y_label_c = y_label_a
    x_label_d = x_label_c
    y_label_d = y_d + height_hist + 0.1
    x_label_e = x_label_c
    y_label_e = y_e + height_hist + 0.1
    x_label_f = x_label_c
    y_label_f = y_f + height_hist + 0.1
    
    #x and y positions of all subfigures and labels IN FRACTIONS OF FIG SIZE
    x_a = (x_a/width_fig)
    y_a = (y_a/height_fig)
    x_a2 = (x_a2/width_fig)
    y_a2 = (y_a2/height_fig)
    x_b = (x_b/width_fig)
    y_b = (y_b/height_fig)
    x_b2 = (x_b2/width_fig)
    y_b2 = (y_b2/height_fig)
    x_c = (x_c/width_fig)
    y_c = (y_c/height_fig)
    x_d = (x_d/width_fig)
    y_d = (y_d/height_fig)
    x_e = (x_e/width_fig)
    y_e = (y_e/height_fig)
    x_f = (x_f/width_fig)
    y_f = (y_f/height_fig)

    
    x_label_a = (x_label_a/width_fig)
    y_label_a = (y_label_a/height_fig)
    x_label_b = (x_label_b/width_fig)
    y_label_b = (y_label_b/height_fig)
    x_label_c = (x_label_c/width_fig)
    y_label_c = (y_label_c/height_fig)
    x_label_d = (x_label_d/width_fig)
    y_label_d = (y_label_d/height_fig)
    x_label_e = (x_label_e/width_fig)
    y_label_e = (y_label_e/height_fig)    
    x_label_f = (x_label_f/width_fig)
    y_label_f = (y_label_f/height_fig)
    
    #widths and heights of all subfigures IN FRACTIONS OF FIG SIZE
    width_traces = (width_traces/width_fig)
    height_traces = (height_traces/height_fig) 
    width_scatter = (width_scatter/width_fig)
    height_scatter = (height_scatter/height_fig) 
    width_hist = (width_hist/width_fig)
    height_hist = (height_hist/height_fig) 
    height_stim = (height_stim/height_fig)
    

    fig = plt.figure(figsize = (width_fig, height_fig), dpi = 300)  
    #fig = plt.figure()
    
    ###################### Panel A: multiple trials, V1 (broadband)
    width = width_traces
    height = height_traces
    x_pos = x_a
    y_pos = y_a
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_a, y_label_a, 'A', fontproperties=fontl)

    ax1 = fig.add_axes(rect)
    
    ###################### Panel A/lower: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_a2
    y_pos = y_a2
    rect = (x_pos, y_pos, width, height)
    ax1_2 = fig.add_axes(rect)
    plot_stim_trace(ax1_2, windows_for_FFT, gaps_for_FFT, padding_for_traces)
    ax1_2.set_xlim(0, sum(windows_for_FFT) + sum(gaps_for_FFT) + 2*padding_for_traces)
    
    ###################### Panel B: multiple trials, V2 (broadband)
    width = width_traces
    height = height_traces
    x_pos = x_b
    y_pos = y_b
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_b, y_label_b, 'B', fontproperties=fontl)

    ax2 = fig.add_axes(rect)
    plot_traces(ax1, ax2, fonts, example_cells_to_plot, 
        windows_for_FFT = windows_for_FFT,
        windows_for_CC = windows_for_CC,
        gaps_for_FFT = gaps_for_FFT, 
        gaps_for_CC = gaps_for_CC, 
        padding = padding_for_traces,
        crit_freq_for_traces = crit_freq_for_traces, 
        filt_kind_for_traces = filt_kind_for_traces,
        freq_range_for_FFT = freq_range_for_FFT,
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)
    
    ax1.set_xlim(0, sum(windows_for_FFT) + sum(gaps_for_FFT) + 2*padding_for_traces)
    ax2.set_xlim(0, sum(windows_for_FFT) + sum(gaps_for_FFT) + 2*padding_for_traces)  
    
    ###################### Panel A/lower: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_b2
    y_pos = y_b2
    rect = (x_pos, y_pos, width, height)
    ax2_2 = fig.add_axes(rect)
    plot_stim_trace(ax2_2, windows_for_FFT, gaps_for_FFT, padding_for_traces)
    ax2_2.set_xlim(0, sum(windows_for_FFT) + sum(gaps_for_FFT) + 2*padding_for_traces)
    
    ###################### Panel C: <FFT> histogram
    width = width_hist
    height = height_hist
    x_pos = x_c
    y_pos = y_c
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_c, y_label_c, 'C', fontproperties=fontl)
    ax3 = fig.add_axes(rect)
    plot_FFT_dist(ax3, fonts, stim_types, windows_for_FFT, gaps_for_FFT,
        padding_for_traces, freq_range_for_FFT, replace_spikes,
        remove_sine_waves, get_new_FFT_results_all_cells = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)


    ###################### Panel D: ong CC vs. <FFT>
    width = width_scatter
    height = height_scatter
    x_pos = x_d
    y_pos = y_d
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_d, y_label_d, 'D', fontproperties=fontl)
    ax4 = fig.add_axes(rect)

    ###################### Panel E: trans CC vs. <FFT>   
    width = width_scatter
    height = height_scatter
    x_pos = x_e
    y_pos = y_e
    rect = (x_pos, y_pos, width, height)
    ax5 = fig.add_axes(rect)
    fig.text(x_label_e, y_label_e, 'E', fontproperties=fontl)

    ###################### Panel F: ss CC vs. <FFT>   
    width = width_scatter
    height = height_scatter
    x_pos = x_f
    y_pos = y_f
    rect = (x_pos, y_pos, width, height)
    ax6 = fig.add_axes(rect)
    fig.text(x_label_f, y_label_f, 'F', fontproperties=fontl)
    
    plot_CC_vs_ongoing_FFT(ax4, ax5, ax6, fonts, stim_types,
        windows_for_FFT, windows_for_CC, gaps_for_FFT, gaps_for_CC,
        padding_for_traces,
        crit_freq_for_CC, filt_kind_for_CC, freq_range_for_FFT,
        replace_spikes, remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_FFT_results_all_cells = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)
   
    
    if savefig == True :
        figpath = master_folder_path + '\\figures\\Fig4'
        
        fig.savefig(figpath + '.png', dpi = 300)      
    else :
        pylab.show()

