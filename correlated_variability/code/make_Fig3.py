
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

from sub_fig_plotting.plot_example_high_freq_residuals_for_cell_pair import \
plot_example_high_freq_residuals_for_cell_pair as plot_resids

from sub_fig_plotting.plot_CC_trajectories import \
plot_CC_trajectories as plot_traj

from sub_fig_plotting.plot_epoch_comparison_CC_scatters import \
plot_epoch_comparison_CC_scatters as plot_scatters

from sub_fig_plotting.plot_CC_PCA_results import \
plot_CC_PCA_results as plot_PCA

from sub_fig_plotting.check_chosen_pair import check_chosen_pair

from sub_fig_plotting.plot_stim_trace import plot_stim_trace

from analysis.compare_CCs_brief_vs_extended_stim import \
compare_CCs_brief_vs_extended_stim as compare_CCs

def make_Fig3(example_cells_to_plot = ['070314_c2', '070314_c3'],
        trial_nums_to_plot = [0, 1, 2, 29],
        windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
        padding_for_traces = 0.0,
        crit_freq = (20.0, 100.0), filt_kind = 'band',
        replace_spikes = True, remove_sine_waves = True,
        master_folder_path = 'F:\\correlated_variability_',
        savefig = False):

    '''
    Parameters
    ----------
    fontproperties: FontProperties object
        dictates size and font of text
    cells_to_plot : python list of strings
        list of cells to plot (e.g., ['070314_c2', '070314_c3'])
    trial_to_plot : python list of ints
        list of trials to plot (e.g., [0, 1, 2, 29]) in panel A
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
    '''

    sub_ong_start = int(padding_for_traces)
    sub_ong_stop = int(windows[0] + padding_for_traces)
    sub_trans_start = int(windows[0] + gaps[0] + gaps[1] + padding_for_traces)
    sub_trans_stop = int(sub_trans_start + windows[1])
    sub_ss_start = int(sub_trans_stop + gaps[2])
    sub_ss_stop = int(sub_ss_start + windows[2])
    
    #width_fig = 7.75 #inches
    width_fig = 8.5
    height_fig = 5.5
    padding = 0.75 #space (in inches) b/w subfigs and edges of figure
    interpadding = 0.6 #reference unit for padding b/w subfigs
    
    fonts = FontProperties(); fonts.set_weight('bold'); fonts.set_size(6)
    fontm = FontProperties(); fontm.set_weight('bold'); fontm.set_size(12)
    fontl = FontProperties(); fontl.set_weight('bold'); fontl.set_size(12)
    
    
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
    side_lower_plot = (1./5.)*(width_fig - 2*padding - 4*interpadding)
    width_scatter = side_lower_plot
    height_scatter = side_lower_plot
    width_traces = 0.5*(width_fig - 2*padding - interpadding)
    width_traj = 0.5*(0.5*(width_fig - 2*padding - interpadding) - interpadding)
    height_stim = 0.2

    width_scatter = side_lower_plot    
    height_scatter = side_lower_plot
    
    width_PCA = side_lower_plot
    height_PCA = side_lower_plot
    
    height_traces = height_fig - 2*padding - interpadding - height_scatter - height_stim
    height_traj = height_traces
        
    
    #x and y positions of all subfigures and labels IN INCHES
    x_a = padding
    y_a = height_fig - padding - height_traces
    x_a2 = padding
    y_a2 = y_a - 0.2
    x_b = width_fig - padding - 2*width_traj - interpadding
    y_b = y_a
    x_c = x_b + interpadding + width_traj
    y_c = y_b
    x_d = x_a
    y_d = padding
    x_e = x_d + interpadding + width_scatter
    y_e = y_d
    x_f = x_e + interpadding + width_scatter
    y_f = y_d
    x_g = width_fig - padding - 2*width_PCA - interpadding
    y_g = y_d
    x_h = width_fig - padding - width_PCA
    y_h = y_d

    
    x_label_a = x_a - 0.4
    y_label_a = y_a + height_traces
    x_label_b = x_b - 0.4
    y_label_b = y_label_a
    x_label_c = x_c - 0.4
    y_label_c = y_label_b
    x_label_d = x_d - 0.4
    y_label_d = y_d + height_scatter
    x_label_e = x_e - 0.4
    y_label_e = y_label_d    
    x_label_f = x_f - 0.4
    y_label_f = y_label_d
    x_label_g = x_g - 0.55
    y_label_g = y_label_d    
    x_label_h = x_h - 0.55
    y_label_h = y_label_d  
    
    #x and y positions of all subfigures and labels IN FRACTIONS OF FIG SIZE
    x_a = (x_a/width_fig)
    y_a = (y_a/height_fig)
    x_a2 = (x_a2/width_fig)
    y_a2 = (y_a2/height_fig)
    x_b = (x_b/width_fig)
    y_b = (y_b/height_fig)
    x_c = (x_c/width_fig)
    y_c = (y_c/height_fig)
    x_d = (x_d/width_fig)
    y_d = (y_d/height_fig)
    x_e = (x_e/width_fig)
    y_e = (y_e/height_fig)
    x_f = (x_f/width_fig)
    y_f = (y_f/height_fig)
    x_g = (x_g/width_fig)
    y_g = (y_g/height_fig)
    x_h = (x_h/width_fig)
    y_h = (y_h/height_fig)
    
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
    x_label_g = (x_label_g/width_fig)
    y_label_g = (y_label_g/height_fig)
    x_label_h = (x_label_h/width_fig)
    y_label_h = (y_label_h/height_fig)
    
    #widths and heights of all subfigures IN FRACTIONS OF FIG SIZE
    width_traces = (width_traces/width_fig)
    height_traces = (height_traces/height_fig) 
    width_traj = (width_traj/width_fig)
    height_traj = (height_traj/height_fig) 
    width_scatter = (width_scatter/width_fig)
    height_scatter = (height_scatter/height_fig) 
    width_PCA = (width_PCA/width_fig)
    height_PCA = (height_PCA/height_fig)
    height_stim = (height_stim/height_fig)
    
    y = (1, 1, 0)
    g = (0, 1, 0)
    b = (0.2, 0.6, 1)
    
    fig = plt.figure(figsize = (width_fig, height_fig), dpi = 300)  
    #fig = plt.figure()
    
    ###################### Panel A: multiple trials, V1-V2 (gamma band)
    width = width_traces
    height = height_traces
    x_pos = x_a
    y_pos = y_a
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_a, y_label_a, 'A', fontproperties=fontl)

    ax1 = fig.add_axes(rect)
    plot_resids(ax1, fonts, example_cells_to_plot, trial_nums_to_plot,
        windows, gaps, padding_for_traces, crit_freq, filt_kind,
        replace_spikes, remove_sine_waves, get_new_processed_traces = False,
        master_folder_path = master_folder_path)
    
    ax1.axvspan(sub_ong_start, sub_ong_stop, ymin = 0, ymax = 1, facecolor = y, 
        ec = 'none', alpha = 0.3)
    ax1.axvspan(sub_trans_start, sub_trans_stop, ymin = 0, ymax = 1, 
        facecolor = b, ec = 'none', alpha = 0.3)
    ax1.axvspan(sub_ss_start, sub_ss_stop, ymin = 0, ymax = 1, facecolor = g, 
        ec = 'none', alpha = 0.3)
    ax1.set_ylim(-20, 3)
    ax1.set_xlim(sub_ong_start, sub_ss_stop)
    
    ###################### Panel A/lower: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_a2
    y_pos = y_a2
    rect = (x_pos, y_pos, width, height)
    ax1_2 = fig.add_axes(rect)
    plot_stim_trace(ax1_2, windows, gaps, padding_for_traces)
    ax1_2.set_xlim(sub_ong_start, sub_ss_stop)
    
    ###################### Panel B: V-V noise correlation trajectories (movies)
    print ''
    print '### CC results ###'
    width = width_traj
    height = height_traj
    x_pos = x_b
    y_pos = y_b
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_b, y_label_b, 'B', fontproperties=fontl)
    ax2 = fig.add_axes(rect)
    print '   EXTENDED STIMULI'
    plot_traj(ax2, fonts, stim_types = ['extended_stim'], 
        windows = windows, gaps = gaps, padding = padding_for_traces,
        crit_freq = crit_freq, filt_kind = filt_kind, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)
    
    ###################### Panel C: V-V noise correlation trajectories (flashes)
    width = width_traj
    height = height_traj
    x_pos = x_c
    y_pos = y_c
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_c, y_label_c, 'C', fontproperties=fontl)
    ax3 = fig.add_axes(rect)
    print '   FLASHES'

    plot_traj(ax3, fonts, stim_types = ['flash'], 
        windows = windows, gaps = gaps, padding = padding_for_traces,
        crit_freq = crit_freq, filt_kind = filt_kind, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)

    ###################### Panel D: V-V trans CC vs. ong CC
    print ''
    print '### epoch CC comparisons ###'
    width = width_scatter
    height = height_scatter
    x_pos = x_d
    y_pos = y_d
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_d, y_label_d, 'D', fontproperties=fontl)
    ax4 = fig.add_axes(rect)

    ###################### Panel E: V-V ss CC vs. trans CC    
    width = width_scatter
    height = height_scatter
    x_pos = x_e
    y_pos = y_e
    rect = (x_pos, y_pos, width, height)
    ax5 = fig.add_axes(rect)
    fig.text(x_label_e, y_label_e, 'E', fontproperties=fontl)

    ###################### Panel F: V-V ss CC vs. ong CC    
    width = width_scatter
    height = height_scatter
    x_pos = x_f
    y_pos = y_f
    rect = (x_pos, y_pos, width, height)
    ax6 = fig.add_axes(rect)
    fig.text(x_label_f, y_label_f, 'F', fontproperties=fontl)
   
    plot_scatters(ax4, ax5, ax6, fonts, stim_types = ['extended_stim', 'flash'],
        windows = windows, gaps = gaps, padding = padding_for_traces,
        crit_freq = crit_freq, filt_kind = filt_kind, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)
    
    ###################### Panel G: variance explained by PC1, PC2, PC3    
    width = width_PCA
    height = height_PCA
    x_pos = x_g
    y_pos = y_g
    rect = (x_pos, y_pos, width, height)
    ax7 = fig.add_axes(rect)
    fig.text(x_label_g, y_label_g, 'G', fontproperties=fontl) 

    ###################### Panel H: variance explained by PC1 for ong, trans, ss    
    width = width_PCA
    height = height_PCA
    x_pos = x_h
    y_pos = y_h
    rect = (x_pos, y_pos, width, height)
    ax8 = fig.add_axes(rect)
    fig.text(x_label_h, y_label_h, 'H', fontproperties=fontl) 
    
    print ''    
    print '### PCA results ###'
    
    plot_PCA(ax7, ax8, fonts, stim_types = ['extended_stim', 'flash'],
        windows = windows, gaps = gaps, padding = padding_for_traces,
        crit_freq = crit_freq, filt_kind = filt_kind, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_processed_traces = False,
        master_folder_path = master_folder_path)

    ###################### compare CCs by stim type (report, but don't plot)
    compare_CCs(windows, gaps,
        padding_for_traces, crit_freq, filt_kind,
        replace_spikes, remove_sine_waves,
        get_new_CC_results_all_pairs = False,
        get_new_processed_traces = False, 
        master_folder_path = master_folder_path)    

    
    if savefig == True :
        figpath = master_folder_path + '\\figures\\Fig3'
        fig.savefig(figpath + '.png', dpi=300)      
    else :
        pylab.show()

