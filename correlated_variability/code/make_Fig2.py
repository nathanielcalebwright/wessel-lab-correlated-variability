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
import urllib2
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from sub_fig_plotting.plot_trials_and_avgs_for_V_V_LFP_triple import \
plot_trials_and_avgs_for_V_V_LFP_triple as plot_trials_V_V_LFP_triple

from sub_fig_plotting.plot_example_trace_with_zoom_inset import \
plot_example_trace_with_zoom_inset as plot_example_trace

from sub_fig_plotting.plot_rel_pwr_spectrum_for_example_cell import \
plot_rel_pwr_spectrum_for_example_cell as plot_rel_pwr_spec

from sub_fig_plotting.plot_rel_pwr_trajectories import \
plot_rel_pwr_trajectories as plot_rP_traj

from sub_fig_plotting.plot_avg_membrane_potential_trajectories import \
plot_avg_membrane_potential_trajectories as plot_V_traj

from sub_fig_plotting.check_chosen_triple import check_chosen_triple

from sub_fig_plotting.plot_stim_trace import plot_stim_trace


def make_Fig2(extended_stim_trodes_to_plot = ['070314_c2', '070314_c3', 
        '070314_LFP2'],
        flash_trodes_to_plot = ['082813_c1', '082813_c2', '082813_LFP1'],
        single_trial_cell = '070314_c3', pwr_spectrum_cell = '070314_c3',
        windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
        padding_for_trace_plots = 0.0, padding_for_pwr = 0.0,
        crit_freq_broadband = 100.0, filt_kind_broadband = 'low',
        crit_freq_pwr_traj = (20.0, 100.0), filt_kind_pwr_traj = 'unfiltered',
        replace_spikes = True, remove_sine_waves = True,
        master_folder_path = 'F:\\correlated_variability_', V_offset = -35,
        LFP_offset = -20, savefig = False):

    """
    Generate Figure 2 from the manuscript.
    
    Parameters
    ----------

    extended_stim_trodes_to_plot : list of strings
        list of simultaneously-recorded trodes to plot in A (extended stim)
        (e.g., ['070314_c2', '070314_c3', '070314_LFP2'])
    flash_trodes_to_plot : list of strings
        list of simultaneously-recorded trodes to plot in B (flash stim)
        (e.g., ['082813_c1', '082813_c2', '082813_LFP1])
    single_trial_cell: string
        cell to use for single-trial example response in D
    pwr_spectrum_cell: string
        cell to use for example residual power spectrum in E
    windows : list of floats
        widths of ongoing, transient, and steady-state windows (ms)
    gaps : list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms)
    padding_for_trace_plots: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace
    padding_for_pwr: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace prior to performing wavelet transform (discarded before 
        calculating power spectrum)
    crit_freq_broadband: float, or tuple of floats
        critical frequency for broad-band filtering of traces corresponding
        to panels A - E (e.g., 100.0)
    filt_kind_broadband: string
        type of filter to be used for traces corresponding to panels A - E
        (e.g., 'low' for lowpass)
    crit_freq_pwr_traj: float, or tuple of floats
        critical frequency for band-pass filtering of traces corresponding
        to panel F (e.g., (20.0, 100.0))
    filt_kind_pwr_traj: string
        type of filter to be used for traces corresponding to panel F
        (e.g., 'band' for bandpass)  
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
    V_offset : float or int
        imposed vertical distance (mV) between traces for one cell and
        traces for the other (in A and B, e.g., -35)
    LFP_offset : float or int
        imposed vertical distance (mV) between traces for one cell and
        traces for the LFP (in A and B, e.g., -20)
    savefig: bool
        if True, save Fig2.png to master_folder_path\\figures\\
        else, show figure

        
    Returns
    -------
    none
    """
    
    fonts = FontProperties(); fonts.set_weight('bold'); fonts.set_size(6)
    fontm = FontProperties(); fontm.set_weight('bold'); fontm.set_size(12)
    fontl = FontProperties(); fontl.set_weight('bold'); fontl.set_size(12)   
    
    #a1:trials & avgs for extended_stim_trodes_to_plot (catcam)
    #a2:movie frames for b
    #a3:stim trace for b
    #b:trials & avgs for flash_trodes_to_plot (640 nm, whole-field)
    #c:<V> trajectories for all cells (movie + flash)
    #d1: single-trial trace for single_trial_cell
    #d2: single-trial trace for single_trial_cell (ZOOM, inset to c1)
    #d3: stim trace for d1
    #e: relative power in residual traces for pwr_spectrum_cell
    #f: power trajectories (for range crit_freq_pwr_traj), 
    #       all cells (movie + flash) 
    
    ### heights and widths (inches) ###    
    width_fig = 8.5
    height_fig = 7.5
    padding = 0.75 #space b/w subfigs and edges of figure
    interpadding = 0.4 #reference unit for padding b/w subfigs
    height_stim_trace = 0.2
    height_movie_trace = 0.4
    height_traces = (3./5.)*(height_fig - padding - 2*interpadding - 2*height_stim_trace)
    width_traces = 0.5*(width_fig - 2*padding - interpadding)
    width_traj = 1.2
    width_mini_traj = (2./3.)*width_traj
    height_traj = (2./5.)*(height_fig - 2*padding - interpadding - height_stim_trace - height_movie_trace)
    height_mini_traj = height_traj
    width_single_trial_trace = (3./5.)*(width_fig - 2*padding - 3*interpadding - width_traj - width_mini_traj) - 0.5*interpadding
    height_single_trial_trace = height_traj - height_stim_trace
    width_spec = (2./5.)*(width_fig - 2*padding - 3*interpadding - width_traj - width_mini_traj)
    height_spec = height_traj

            
    width_a1 = width_traces
    height_a1 = height_traces
    width_a2 = width_traces
    height_a2 = height_movie_trace
    width_a3 = width_traces
    height_a3 = height_stim_trace
    width_b = width_a1
    height_b = height_a1    
    
    width_c = width_traj
    height_c = height_traj
    width_d1 = width_single_trial_trace
    height_d1 =  height_single_trial_trace
    width_d2 = 0.4*width_d1
    height_d2 = 0.75*height_d1
    width_d3 = width_d1
    height_d3 = height_stim_trace
    width_e = width_spec
    height_e = height_spec
    width_f = width_mini_traj
    height_f = height_mini_traj
    
    #x and y positions of all subfigures and labels IN INCHES
    x_a1 = padding
    y_a1 = height_fig - padding - height_a1
    x_a3 = x_a1
    y_a3 = y_a1 - 1.5*height_a3
    x_a2 = x_a1
    y_a2 = y_a3 + height_stim_trace
    x_b = width_fig - width_b - padding
    y_b = y_a1
    
    x_c = padding
    y_c = padding
    x_d1 = x_c + width_traj + 0.75*interpadding
    y_d1 = padding + height_stim_trace
    x_d2 = x_d1
    y_d2 = y_d1 + height_d1 - 0.8*height_d2
    x_d3 = x_d1
    y_d3 = padding
    x_e = x_d1 + width_single_trial_trace + 1.25*interpadding
    y_e = padding
    x_f = width_fig - padding - width_mini_traj
    y_f = padding
    
    x_label_a = x_a1 - 0.2
    y_label_a = y_a1 + height_a1
    x_label_b = x_b - 0.2
    y_label_b = y_label_a
    x_label_c = x_c - 0.2
    y_label_c = y_c + height_traj
    x_label_d = x_d1 - 0.2
    y_label_d = y_label_c
    x_label_e = x_e - 0.35
    y_label_e = y_label_d
    x_label_f = x_f - 0.35
    y_label_f = y_f + height_f
        
    #x and y positions of all subfigures and labels IN FRACTIONS OF FIG SIZE    
    x_a1 = (x_a1/width_fig)
    y_a1 = (y_a1/height_fig)
    x_a2 = (x_a2/width_fig)
    y_a2 = (y_a2/height_fig)
    x_a3 = (x_a3/width_fig)
    y_a3 = (y_a3/height_fig)
    
    x_b = (x_b/width_fig)
    y_b = (y_b/height_fig)
    x_c = (x_c/width_fig)
    y_c = (y_c/height_fig)    
    x_d1 = (x_d1/width_fig)
    y_d1 = (y_d1/height_fig)
    x_d2 = (x_d2/width_fig)
    y_d2 = (y_d2/height_fig)
    x_d3 = (x_d3/width_fig)
    y_d3 = (y_d3/height_fig)
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
    width_a1 = (width_a1/width_fig)
    height_a1 = (height_a1/height_fig)
    width_a2 = width_a2/width_fig
    height_a2 = height_a2/height_fig
    
    #height_a2 = (height_a2/height_fig)
    width_a3 = (width_a3/width_fig)
    height_a3 = (height_a3/height_fig)
    
    width_b = (width_b/width_fig)
    height_b =  (height_b/height_fig)
    
    width_c = (width_c/width_fig)
    height_c = (height_c/height_fig)
    width_d1 = (width_d1/width_fig)
    height_d1 = (height_d1/height_fig)
    width_d2 = (width_d2/width_fig)
    height_d2 = (height_d2/height_fig)
    width_d3 = (width_d3/width_fig)
    height_d3 = (height_d3/height_fig)
    width_e = (width_e/width_fig)
    height_e = (height_e/height_fig)
    width_f = (width_f/width_fig)
    height_f = (height_f/height_fig)
    
    #fig = plt.figure()
    fig = plt.figure(figsize = (width_fig, height_fig), dpi = 300)    
    
    ###################### Panel A: trials & avgs for extended_stim_trodes_to_plot (catcam)    
    width = width_a1
    height = height_a1
    x_pos = x_a1
    y_pos = y_a1
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_a, y_label_a, 'A', fontproperties=fontl)
    ax1 = fig.add_axes(rect)
    plot_decision = check_chosen_triple('extended_stim', 
        extended_stim_trodes_to_plot, master_folder_path)
    if plot_decision == True:
        plot_trials_V_V_LFP_triple(ax1, fonts, extended_stim_trodes_to_plot,
            windows, gaps, padding_for_trace_plots, crit_freq_broadband, 
            filt_kind_broadband, V_offset = V_offset, LFP_offset = LFP_offset,
            replace_spikes = replace_spikes, 
            remove_sine_waves = remove_sine_waves, mult_for_LFP = 150.0,
            master_folder_path = master_folder_path)    
        ax1.set_xlim(padding_for_trace_plots, 
        padding_for_trace_plots + sum(windows) + sum(gaps))
        #ax1.set_ylim(-50, 35)
    
    ###################### Panel A2: movie frames
    width = width_a2
    height = height_a2
    x_pos = x_a2
    y_pos = y_a2
    rect = (x_pos, y_pos, width, height)
    axa_2 = fig.add_axes(rect)
    image_path = 'file:\\' + master_folder_path + '\\figures\\catcam_frames.png'
    image = urllib2.urlopen(image_path)
    array = pylab.imread(image)
    axa_2.imshow(array)    
    axa_2.set_axis_off()
    
    ###################### Panel A3: stim trace
    width = width_a3
    height = height_a3
    x_pos = x_a3
    y_pos = y_a3
    rect = (x_pos, y_pos, width, height)
    ax1_3 = fig.add_axes(rect)
    plot_stim_trace(ax1_3, windows, gaps, padding)
    ax1_3.set_xlim(padding_for_trace_plots, 
        padding_for_trace_plots + sum(windows) + sum(gaps))
    
    
    ###################### Panel B: trials & avgs for flash_trodes_to_plot (640 nm, whole-field)
    width = width_b
    height = height_b
    x_pos = x_b
    y_pos = y_b
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_b, y_label_b, 'B', fontproperties=fontl)
    ax2 = fig.add_axes(rect)
    plot_decision = check_chosen_triple('flash', 
        flash_trodes_to_plot, master_folder_path)
    if plot_decision == True:
        plot_trials_V_V_LFP_triple(ax2, fonts, flash_trodes_to_plot,
            windows, gaps, padding_for_trace_plots, crit_freq_broadband, 
            filt_kind_broadband, V_offset = V_offset, LFP_offset = LFP_offset,
            replace_spikes = replace_spikes, 
            remove_sine_waves = remove_sine_waves, mult_for_LFP = 500.0,
            label_scale_bars = False,
            master_folder_path = master_folder_path)

        ax2.set_xlim(padding_for_trace_plots, 
            padding_for_trace_plots + sum(windows) + sum(gaps))
        #ax2.set_ylim(-50, 35)
    
    ### draw arrow at stim onset ###
    ax2.arrow(padding + windows[0] + gaps[0], -80, 0, 8, head_width = 100, 
        head_length = 3, overhang = 0.5, fc = 'm', ec = 'm')

    ###################### Panel C: <V> trajectories for all cells (movie + flash)

    print '### <V> results ###'
    width = width_c
    height = height_c
    x_pos = x_c
    y_pos = y_c
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_c, y_label_c, 'C', fontproperties=fontl)
    ax3 = fig.add_axes(rect)
    plot_V_traj(ax3, fonts, stim_types = ['extended_stim', 'flash'],
        windows = windows, gaps = gaps, padding = padding_for_trace_plots, 
        crit_freq = crit_freq_broadband, filt_kind = filt_kind_broadband, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        master_folder_path = master_folder_path)
   
    ###################### Panel D1: single-trial trace for single_trial_cell
    width = width_d1
    height = height_d1
    x_pos = x_d1
    y_pos = y_d1
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_d, y_label_d, 'D', fontproperties=fontl)
    ax4 = fig.add_axes(rect)
    
    ###################### Panel D2: single-trial trace for single_trial_cell (ZOOM)
    width = width_d2
    height = height_d2
    x_pos = x_d2
    y_pos = y_d2
    rect = (x_pos, y_pos, width, height)
    ax5 = fig.add_axes(rect)
    plot_example_trace(ax4, ax5, fonts, single_trial_cell, trial_num = 2,
        windows = windows, gaps = gaps, padding = padding_for_trace_plots, 
        crit_freq = crit_freq_broadband, filt_kind = filt_kind_broadband, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        master_folder_path = master_folder_path)
  
    ###################### Panel D3: stim trace for D1
    width = width_d3
    height = height_d3
    x_pos = x_d3
    y_pos = y_d3
    rect = (x_pos, y_pos, width, height)
    ax6 = fig.add_axes(rect)
    plot_stim_trace(ax6, windows, gaps, padding_for_trace_plots)
    ax6.set_xlim(padding, padding + sum(windows) + sum(gaps))
    
    ###################### Panel E: relative power in residual traces 
    ###################### for pwr_spectrum_cell
    width = width_e
    height = height_e
    x_pos = x_e
    y_pos = y_e
    rect = (x_pos, y_pos, width, height)
    ax7 = fig.add_axes(rect)
    fig.text(x_label_e, y_label_e, 'E', fontproperties=fontl)
    plot_rel_pwr_spec(ax7, fonts, pwr_spectrum_cell, 
        windows, gaps, padding_for_pwr, crit_freq_broadband, replace_spikes, 
        remove_sine_waves,
        get_new_processed_traces = False, get_new_pwr_dict = False,
        master_folder_path = master_folder_path)    
    
    ###################### Panel F: power trajectories 
    ###################### (for range crit_freq_pwr_traj) for population
    print '### <rP> results ###'
    width = width_f
    height = height_f
    x_pos = x_f
    y_pos = y_f
    rect = (x_pos, y_pos, width, height)
    ax8 = fig.add_axes(rect)
    fig.text(x_label_f, y_label_f, 'F', fontproperties=fontl)
    plot_rP_traj(ax8, fonts, stim_types = ['extended_stim', 'flash'],
        windows = windows, gaps = gaps, padding = padding_for_pwr, 
        crit_freq = crit_freq_pwr_traj, 
        replace_spikes = replace_spikes, remove_sine_waves = remove_sine_waves,
        master_folder_path = master_folder_path)
    
    
    if savefig == True :
        figpath = master_folder_path + '\\figures\\Fig2'
        fig.savefig(figpath + '.png', dpi=300)    
    else :
        pylab.show()

