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
from scipy.stats import wilcoxon as wlcx
import os
import cPickle as pickle
import matplotlib.ticker as plticker

from analysis.get_avg_membrane_potential_all_cells import \
get_avg_membrane_potential_all_cells as get_V_dict_by_cell_all_trials

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

g = (0, 1, 0)
b = (0.2, 0.6, 1)

def plot_trajectory(ax, fontproperties, V_dict, boots_dict, 
        freq_band_for_plot):    
    
    '''
    Plot the trial-averaged membrane potentials in 'trajectory'
    form.  Each dot indicates the across-trial average membrane potential for
    one cell, for the indicated epoch. Lines connect dots for individual cells 
    across epochs.  Lines have higher opacity and lineweight if change is 
    significant (i.e., bootstrap bands from the two epochs do not overlap).  
    Line and label across top indicate results of test for significant change 
    in population-average value across epochs (Wilcoxon signed-rank test, with 
    higher-opacity line indicating p < 0.05, and asterisks used to indicate 
    value of p if significant).
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    V_dict: python dictionary
        dictionary containing across-trial average V values for each cell,
        and bootstrap ranges.
    freq_band_for_plot: tuple of floats
        range of frequencies considered in Hz (e.g., (20.0, 100.0))
        
    Returns
    -------
    none    
    '''

    fonts = fontproperties
    size = fonts.get_size()
    markersize = 6

    y = (1, 1, 0) #colors for dots
    g = (0, 1, 0)
    b = (0.2, 0.6, 1)
        
    ong_params = V_dict['ongoing']
    ong_boots = boots_dict['ongoing']
    trans_params = V_dict['transient']
    trans_boots = boots_dict['transient']
    ss_params = V_dict['steady-state']
    ss_boots = boots_dict['steady-state']  
    
    #need left-right "jitter" to make overlapping dots distinguishable
    jitters = numpy.zeros(len(trans_params))
    for n, val in enumerate(jitters):
        jitters[n] += numpy.random.uniform(-0.05, 0.05)
    
    ### Get p-vals for stim modulation of population, draw on plot.
    ong_trans_stat, ong_trans_p_val = wlcx(ong_params, trans_params)    
    trans_ss_stat, trans_ss_p_val = wlcx(trans_params, ss_params)
    ong_ss_stat, ong_ss_p_val = wlcx(ong_params, ss_params)
    
    #correct p-vals for multiple comparisons
    ong_trans_p_val *= 1./3.
    trans_ss_p_val *= 1./3.
    ong_ss_p_val *= 1./3.

    print '  <V> ong = %s'%str(numpy.mean(ong_params)), '+/- %s (mV; mean +/- s.e.m.)'%str(numpy.std(ong_params)) 
    print '  <V> trans = %s'%str(numpy.mean(trans_params)), '+/- %s (mV; mean +/- s.e.m.)'%str(numpy.std(trans_params)) 
    print '  <V> ss = %s'%str(numpy.mean(ss_params)), '+/- %s (mV; mean +/- s.e.m.)'%str(numpy.std(ss_params))   
    print '  ong --> trans P_corrected = %s'%str(ong_trans_p_val), '(stat = %s)'%str(ong_trans_stat)
    print '  trans --> ss P_corrected = %s'%str(trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)
    print '  ong --> ss P_corrected = %s'%str(ong_ss_p_val), '(stat = %s)'%str(ong_ss_stat)

    y_offset = 0.5
    y1 = max(max(ong_params), max(trans_params), max(ss_params)) + 0.02
    y2 = y1 + 3.5

    if ong_trans_p_val < 0.05:
        if 0.001 <= ong_trans_p_val < 0.01:
            ong_trans_p_for_plot = '**'
        elif ong_trans_p_val < 0.001:
            ong_trans_p_for_plot = '***'                
        else:
            ong_trans_p_for_plot = '*'
        ax.plot([-0.1, 0.75], [y1, y1], color = 'k', lw = 2.0)

        
    else:
        x_pos = 0.325
        ong_trans_p_for_plot = ''
        ax.plot([-0.1, 0.75], [y1, y1], color = 'k', alpha = 0.35, lw = 2.0)

    x_pos = 0.325
    ax.text(x_pos, y1 + y_offset, ong_trans_p_for_plot, fontsize = size, 
        horizontalalignment = 'center')
        
    if trans_ss_p_val < 0.05:
        if 0.001 <= trans_ss_p_val < 0.01:
            trans_ss_p_for_plot = '**'
        elif trans_ss_p_val < 0.001:
            trans_ss_p_for_plot = '***'                
        else:
            trans_ss_p_for_plot = '*'

        ax.plot([1.3, 2.1], [y1, y1], color = 'k', lw = 2.0)
        
    else:
        trans_ss_p_for_plot = ''
        ax.plot([1.3, 2.1], [y1, y1], color = 'k', alpha = 0.35, lw = 2.0)

    x_pos = 1.7
    ax.text(x_pos, y1 + y_offset, trans_ss_p_for_plot, fontsize = size, 
        horizontalalignment = 'center')
        
    if ong_ss_p_val < 0.05:
        if 0.001 <= ong_ss_p_val < 0.01:
            ong_ss_p_for_plot = '**'
        elif ong_ss_p_val < 0.001:
            ong_ss_p_for_plot = '***'                
        else:
            ong_ss_p_for_plot = '*'
        ax.plot([-0.1, 2.1], [y2, y2], color = 'k', lw = 2.0)


    else:
        x_pos = 1.0
        ong_ss_p_for_plot = ''
        ax.plot([-0.1, 2.1], [y2, y2], color = 'k', alpha = 0.35, lw = 2.0)
    
    x_pos = 1.0
    ax.text(x_pos, y2 + y_offset, ong_ss_p_for_plot, fontsize = size, 
        horizontalalignment = 'center')

        
    ### draw lines connecting markers across epochs for each pair,
    ### with line style determined by significance of change across
    ### epoch transition

    for i, ent in enumerate(trans_boots):
        trans_ong_overlap = False
        for entry in trans_boots[i]:
            if ong_boots[i][0] <= entry <= ong_boots[i][1]:
                trans_ong_overlap = True
                break
        for entry in ong_boots[i]:
            if trans_boots[i][0] <= entry <= trans_boots[i][1]:
                trans_ong_overlap = True
                break
        if trans_ong_overlap:
            ls = '-'
            alpha = 0.4
            lw = 0.7
        else:
            ls = '-'
            alpha = 1.0
            lw = 1.0
        ax.plot([0 + jitters[i], 1 + jitters[i]], [ong_params[i], trans_params[i]],
            color = 'k', ls = ls, lw = lw, alpha = alpha)
        
        ss_trans_overlap = False
        for entry in ss_boots[i]:
            if trans_boots[i][0] <= entry <= trans_boots[i][1]:
                ss_trans_overlap = True
                break
        for entry in trans_boots[i]:
            if ss_boots[i][0] <= entry <= ss_boots[i][1]:
                ss_trans_overlap = True
                break                
        if ss_trans_overlap:
            ls = '-'
            alpha = 0.4
            lw = 0.7
        else:
            ls = '-'
            alpha = 1.0
            lw = 1.0
        ax.plot([1 + jitters[i], 2 + jitters[i]], [trans_params[i], ss_params[i]],
            color = 'k', ls = ls, lw = lw, alpha = alpha)

    ### Plot points for each pair ### .

    #ongoing
    for q, entry in enumerate(ong_params):
        ong_mfc = y
        ong_alpha = 1.0
        ax.plot(0 + jitters[q], ong_params[q], 'o', mec = 'k',
            mfc = ong_mfc, mew = 0.5, alpha = ong_alpha, 
            markersize = markersize, clip_on = False)

    #transient
    for c, val in enumerate(trans_params):
        trans_mfc = b
        trans_alpha = 1.0
        ax.plot(1 + jitters[c], trans_params[c], 'o', mec = 'k',
            mfc = trans_mfc, mew = 0.5, alpha = trans_alpha,
            markersize = markersize, clip_on = False)
        
    #steady-state
    for d, val in enumerate(ss_params):
        ss_mfc = g
        ss_alpha = 1.0
        ax.plot(2 + jitters[d], ss_params[d], 'o', mec = 'k',
            mfc = ss_mfc, mew = 0.5, alpha = ss_alpha,
            markersize = markersize, clip_on = False)

    ### configure the plot ###
    ax.set_xlim(-0.2, 2.2)
    loc = plticker.MultipleLocator(base = 20)
    ax.yaxis.set_major_locator(loc)
    #ax.locator_params(axis = 'y', nbins = 4)

        
    ax.set_xticks([0, 1, 2])
    labels = ['ongoing','transient', 'steady-\nstate']
    ax.set_xticklabels(labels, size = size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size) 
    #ax.tick_params('both', length = 5.5, width = 1.3, which = 'major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    yaxis_label = r'$\overline{V}$ (mV)'
    ax.set_ylabel(yaxis_label, fontsize = size)
            

def plot_avg_membrane_potential_trajectories(ax, fontproperties,
        stim_types = ['extended_stim', 'flash'],
        windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
        padding = 0.0, crit_freq = 100.0, filt_kind = 'low',
        replace_spikes = True, remove_sine_waves = True,
        get_new_V_results_all_cells = False,
        get_new_processed_traces = False, 
        master_folder_path = 'E:\\correlated_variability'):
        
    """
    Calculate the trial-averaged membrane potential for each
    cell in the dataset, for the transient and steady-state epochs.  
    Plot the results in "trajectory" view.  For each
    cell, the significance of the across-epoch change is assessed by
    bootstrapping.  For the population, the significance of the across-epoch
    change is assessed via the Wilcoxon signed-rank test.
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    stim_types: python list of strings
        types of visual stimuli to include in this analysis (e.g.,
        ['extended_stim', 'flash'])
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
        frequency range over which to average (e.g., (20.0, 100.0))
    filt_kind: string
        type of filter to use (e.g., 'low' for lowpass)
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via inteVolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    get_new_V_results_all_cells: bool
        if True, get the subtraces, calcualte <V> in each epoch.  otherwise, 
        use saved results if they exist, to speed things up.
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
    
    epochs = ['ongoing', 'transient', 'steady-state']

    V_dict = {}
    bootstrap_ranges = {}
    
    for epoch in epochs:
        V_dict[epoch] = []
        bootstrap_ranges[epoch] = []               

    ### populate the dictionaries ###
    V_dict_by_cell_all_trials = get_V_dict_by_cell_all_trials(stim_types,
        windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
        remove_sine_waves, get_new_V_results_all_cells,
        get_new_processed_traces, master_folder_path) 
    
    #avg_V_dict_by_cell_all_trials = {cell_name:{'ongoing':[avg_V_trial1, 
    #    ..., avg_V_trialn], ..., 'steady-state':[avg_V_trial1,
    #    ..., avg_V_trialn]]}}

    for cell_name in V_dict_by_cell_all_trials:

        avg_V_dict_for_cell = V_dict_by_cell_all_trials[cell_name]
        for epoch in epochs:
            avg_V_all_trials = avg_V_dict_for_cell[epoch]
                
            ## get bootstraps
            bootstrap = get_bootstrap(avg_V_all_trials, num_reps = 1000,
                conf_level = 0.95, num_comparisons = 1) 
            
            ## add trial-avg V and bootstrap to V_dict
            V_dict[epoch].append(numpy.mean(avg_V_all_trials))
            bootstrap_ranges[epoch].append(bootstrap)
      

    
    ### report number of cells and preparations ###
    experiments = []
    for cell_name in V_dict_by_cell_all_trials:
        expt_date = cell_name.split('_')[0]
        if expt_date not in experiments:
            experiments.append(expt_date)
    print '  %s cells;'%str(len(V_dict_by_cell_all_trials.keys())), \
          '%s turtles'%str(len(experiments))
    
    plot_trajectory(ax, fontproperties, V_dict, bootstrap_ranges, 
        freq_band_for_plot = crit_freq)

