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
from scipy.stats import wilcoxon as wlcx, ttest_1samp
import os
import cPickle as pickle
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.get_power_spectra_all_cells import \
get_power_spectra_all_cells as get_pwr_spec_dict_by_cell_all_trials

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

g = (0, 1, 0)
b = (0.2, 0.6, 1)
colors = [b, g]

def plot_trajectory(ax, fontproperties, rP_dict, boots_dict, 
        freq_band_for_plot):    
    
    '''
    Plot the trial-averaged relative power (evoked/ongoing) in 'trajectory'
    form.  Each dot indicates the across-trial average relative power for
    one cell, for the indicated evoked epoch.  Dot is filled if value is
    significant (i.e., bootstrap bands do not overlap rP = 1), empty otherwise.
    Lines connect dots for individual cells across epochs.  Lines have higher
    opacity and lineweight if change is significant (i.e., bootstrap bands from
    the two epochs do not overlap).  Line and label across top indicate results
    of test for significant change in population-average value across epochs
    (Wilcoxon signed-rank test, with higher-opacity line indicating p < 0.05,
    and asterisks used to indicate value of p if significant).
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    rP_dict: python dictionary
        dictionary containing across-trial average rP values for each cell,
        and bootstrap ranges.
    freq_band_for_plot: tuple of floats
        range of frequencies considered in Hz (e.g., (20.0, 100.0))
        
    Returns
    -------
    none    
    '''

    fonts = fontproperties
    size = fonts.get_size()
    mpl.rcParams['mathtext.default'] = 'regular'
    
    if type(freq_band_for_plot) == float:
        freq_band_label = 'rP (0 - %s Hz)'%str(freq_band_for_plot)
    else:
        freq_band_label = r'$\ rP_{hf}$ (%s - '%str(int(freq_band_for_plot[0])) + '%s Hz)'%str(int(freq_band_for_plot[1]))
    y_offset = 2
    y1 = 190
    ax.set_ylabel(freq_band_label, fontsize = size)
        
    trans_params = rP_dict['transient']
    trans_boots = boots_dict['transient']
    ss_params = rP_dict['steady-state']
    ss_boots = boots_dict['steady-state']  
    
    #need left-right "jitter" to make overlapping dots distinguishable
    jitters = numpy.zeros(len(trans_params))
    for n, val in enumerate(jitters):
        jitters[n] += numpy.random.uniform(-0.05, 0.05)
    
    ### Get p-vals for stim modulation of population, draw on plot.
    
    trans_ss_stat, trans_ss_p_val = wlcx(trans_params, ss_params)

    print '  <rP> trans = %s'%str(numpy.mean(trans_params)), '+/- %s (mV; mean +/- s.e.m.)'%str(numpy.std(trans_params))
    print '    (P = %s; one-sided t-test)'%str(0.5*ttest_1samp(trans_params, 0)[1])
    print '  <rP> ss = %s'%str(numpy.mean(ss_params)), '+/- %s (mV; mean +/- s.e.m.)'%str(numpy.std(ss_params))
    print '    (P = %s; one-sided t-test)'%str(0.5*ttest_1samp(ss_params, 0)[1])    
    print '  trans --> ss P = %s'%str(trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)

    if trans_ss_p_val < 0.05:
        if 0.001 <= trans_ss_p_val < 0.01:
            trans_ss_p_for_plot = '**'
        elif trans_ss_p_val < 0.001:
            trans_ss_p_for_plot = '***'                
        else:
            trans_ss_p_for_plot = '*'
        
        ax.plot([0.1, 0.9], [y1, y1], color = 'k', lw = 1.5)
            
    else:    
        trans_ss_p_for_plot = ''
        ax.plot([0.1, 0.9], [y1, y1], color = 'k', alpha = 0.35, lw = 1.5)
        
    x_pos = 0.5
    ax.text(x_pos, y1 + y_offset, trans_ss_p_for_plot, fontsize = 8, 
        horizontalalignment = 'center')

    
    for n, val in enumerate(trans_params):
        ### truncate one outlier value for the plot ###
        if trans_params[n] > 200:
            print '    truncated the value rP = %s (transient epoch) for plot'%str(trans_params[n])
            trans_params[n] += 200 - val

    ### draw lines connecting markers across epochs for each trode,
    ### with line style determined by significance of change across
    ### epoch transition

    for i, ent in enumerate(trans_boots):        
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
            lw = 0.4
        else:
            ls = '-'
            alpha = 0.7
            lw = 0.7
        ax.plot([0 + jitters[i], 1 + jitters[i]], [trans_params[i], ss_params[i]],
            color = 'k', ls = ls, lw = lw, alpha = alpha)

    ### Plot points for each trode (with fill based on significance relative
    ### to rP = 1.0).

    #transient
    for c, val in enumerate(trans_params):
        trans_one_overlap = False
        if trans_boots[c][0] <= 1.0 <= trans_boots[c][1]:
            trans_one_overlap = True
        if trans_one_overlap:
            trans_mfc = (1, 1, 1)
            trans_alpha = 1.0
        else:
            trans_mfc = b
            trans_alpha = 1.0
        ax.plot(0 + jitters[c], trans_params[c], 'o', mec = 'k',
            mfc = trans_mfc, mew = 0.75, alpha = trans_alpha, markersize = 6,
            clip_on = False)
        
    #steady-state
    for d, val in enumerate(ss_params):
        ss_one_overlap = False
        if ss_boots[d][0] <= 1.0 <= ss_boots[d][1]:
            ss_one_overlap = True
        if ss_one_overlap:
            ss_mfc = (1, 1, 1)
            ss_alpha = 1.0
        else:
            ss_mfc = g
            ss_alpha = 1.0
        ax.plot(1 + jitters[d], ss_params[d], 'o', mec = 'k',
            mfc = ss_mfc, mew = 0.75, alpha = ss_alpha, markersize = 6,
            clip_on = False)

    ### configure the plot ###
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    x_labels = ['transient', 'steady-\nstate']
    ax.set_xticklabels(x_labels, size = size)
    
    ### Warning!!! These axis labels are set manually for the final publication
    ### figure.  If parameters are changed, this section should be
    ### commented out.
    y_labels = ['0', '50', '100', '150', '>200']
    ax.set_yticklabels(y_labels, rotation = 'vertical', size = size)

    ax.tick_params('both', length = 4, width = 1.0, which = 'major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
            

def plot_rel_pwr_trajectories(ax, fontproperties,
        stim_types = ['extended_stim', 'flash'],
        windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
        padding = 0.0, crit_freq = (20, 100.0), 
        replace_spikes = True, remove_sine_waves = True,
        get_new_pwr_results_all_cells = False,
        get_new_processed_traces = False, 
        master_folder_path = 'E:\\correlated_variability'):
    
    """
    Calculate the trial-averaged relative power (evoked/ongoing) for each
    cell in the dataset, for the transient and steady-state epochs, for the
    20 - 100 Hz freq band.  Plot the results in "trajectory" view.  For each
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
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    get_new_pwr_results_all_cells: bool
        if True, get the subtraces, do the wavelet transform, etc.  otherwise, 
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
        
    rP_dict = {}
    bootstrap_ranges = {}
    
    for epoch in epochs:
        rP_dict[epoch] = []
        bootstrap_ranges[epoch] = []               

    if type(crit_freq) == tuple:
        hi_freq = crit_freq[1]
    else:
        hi_freq = crit_freq
    
    ### populate the dictionaries ###
    pwr_spec_dict_by_cell_all_trials = get_pwr_spec_dict_by_cell_all_trials(stim_types,
        windows, gaps, padding, hi_freq, replace_spikes, 
        remove_sine_waves, get_new_pwr_results_all_cells,
        get_new_processed_traces, master_folder_path) 
    
    #pwr_spec_dict_by_cell_all_trials = {cell_name:{'ongoing':{'pwr':[[pwr_spec_trial1], 
    #    ..., [pwr_spec_trialn]]}, 'freqs':[[freqs_trial1], ... ], ..., }}

    for cell_name in pwr_spec_dict_by_cell_all_trials:
        pwr_spec_dict_for_cell = pwr_spec_dict_by_cell_all_trials[cell_name]
        full_ong_pwr_spec_all_trials = pwr_spec_dict_for_cell['ongoing']['pwr']
        #get pwr for freqs of interest
        ong_pwr_spec_all_trials = []
        if type(crit_freq) != tuple:
            low_ndx = 0
            hi_ndx = int(crit_freq)
        else:
            low_ndx = crit_freq[0]
            hi_ndx = crit_freq[1]
        for full_ong_spec in full_ong_pwr_spec_all_trials:
            ong_pwr_spec_all_trials.append(full_ong_spec[low_ndx:hi_ndx])
        ong_pwr_spec_all_trials = pylab.array(ong_pwr_spec_all_trials)
        for epoch in epochs[1:]:
            full_pwr_spec_all_trials = pwr_spec_dict_for_cell[epoch]['pwr']
            full_freqs = pwr_spec_dict_for_cell[epoch]['freqs'][0]
            pwr_spec_all_trials = []
            freqs = []
            for full_spec in full_pwr_spec_all_trials:
                pwr_spec_all_trials.append(full_spec[low_ndx:hi_ndx])
                freqs.append(full_freqs[low_ndx:hi_ndx])
            pwr_spec_all_trials = pylab.array(pwr_spec_all_trials)                
            ## get relative power for each freq and trial
            rP_spec_all_trials = pwr_spec_all_trials/ong_pwr_spec_all_trials
            
            ## average over freqs of interest
            rPs_all_trials = []
            for rP_spec in rP_spec_all_trials:
                #print len(rP_spec)
                total_rP = 0
                num_freqs = 0
                #only consider freqs in crit_freq range    
                if type(crit_freq) == float:
                    low_freq = 0
                    hi_freq = crit_freq
                else:
                    low_freq = crit_freq[0]
                    hi_freq = crit_freq[1]
                for ndx, rP in enumerate(rP_spec):
                    if low_freq <= ndx <= hi_freq:
                        total_rP += rP
                        num_freqs += 1
                mean_rP = float(total_rP)/float(num_freqs)
                
                rPs_all_trials.append(mean_rP)
                                        
            ## get bootstraps
            bootstrap = get_bootstrap(rPs_all_trials, num_reps = 1000,
                conf_level = 0.95, num_comparisons = 1) 
            
            ## add trial-avg rP and bootstrap to rP_dict
            rP_dict[epoch].append(numpy.mean(rPs_all_trials))
            bootstrap_ranges[epoch].append(bootstrap)
  

    ### report number of cells and preparations ###
    experiments = []
    for cell_name in pwr_spec_dict_by_cell_all_trials:
        expt_date = cell_name.split('_')[0]
        if expt_date not in experiments:
            experiments.append(expt_date)
    print '    %s cells;'%str(len(pwr_spec_dict_by_cell_all_trials.keys())), \
          '%s turtles'%str(len(experiments))
    
    plot_trajectory(ax, fontproperties, rP_dict, bootstrap_ranges, 
        freq_band_for_plot = crit_freq)

