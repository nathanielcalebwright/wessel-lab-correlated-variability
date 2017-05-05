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
import matplotlib.ticker as plticker
import math

from analysis.get_power_spectra_all_cells import \
get_power_spectra_all_cells as get_pwr_spec_dict_by_cell_all_trials


def plot_grand_avg_rel_pwr_spectra(ax1, ax2, fontproperties,
        stim_types = ['extended_stim', 'flash'],
        windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
        padding = 0.0, crit_freq = 100.0, 
        replace_spikes = True, remove_sine_waves = True,
        get_new_pwr_results_all_cells = False,
        get_new_processed_traces = False, 
        master_folder_path = 'E:\\correlated_variability'):
    
    """
    Calculate the trial-averaged relative power spectrum (evoked/ongoing) for 
    each cell in the dataset, for the transient and steady-state epochs, and
    then average across all cells.
    
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
    crit_freq: float
        defines freq range of interest (e.g., 100 Hz --> inspect 0 - 100 Hz)
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    get_new_pwr_results_all_cells: bool
        if True, get the subtraces, do the power spectral analysis, etc.  
        otherwise, use saved results if they exist, to speed things up.
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
    epochs = ['transient', 'steady-state']
        
    
    ### rP_dict format ###
    
    # rP_dict = {'transient':[[rP_c1], [rP_c2], ... ], ... , 
    #    'steady-state':[rP_c1, ... ]}  where rP_ci is across-trial avg. spec
    
    rP_dict = {}
    freqs = []
    b = (0.2, 0.6, 1)
    g = (0, 1, 0)
    
    for epoch in epochs:
        rP_dict[epoch] = []

    if type(crit_freq) == tuple:
        hi_freq = crit_freq[1]
    else:
        hi_freq = crit_freq
    
    ### populate the dictionaries ###
    pwr_spec_dict_by_cell_all_trials = get_pwr_spec_dict_by_cell_all_trials(stim_types,
        windows, gaps, padding, hi_freq, replace_spikes, 
        remove_sine_waves, get_new_pwr_results_all_cells,
        get_new_processed_traces, master_folder_path) 
    
    #pwr_spec_dict_by_cell_all_trials = {'pwr':{cell_name:{'ongoing':[[pwr_spec_trial1], 
    #    ..., [pwr_spec_trialn]], ..., 'steady-state':[[pwr_spec_trial1,
    #    ..., [pwr_spec_trialn]]]}}, 'freqs':[freq_list]}

    for cell_name in pwr_spec_dict_by_cell_all_trials:
        #plt.figure()

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
        for epoch in epochs:
            full_pwr_spec_all_trials = pwr_spec_dict_for_cell[epoch]['pwr']
            full_freqs = pwr_spec_dict_for_cell[epoch]['freqs'][0]
            pwr_spec_all_trials = []
            freqs = []
            for full_spec in full_pwr_spec_all_trials:
                pwr_spec_all_trials.append(full_spec[low_ndx:hi_ndx])
                freqs = full_freqs[low_ndx:hi_ndx]
            pwr_spec_all_trials = pylab.array(pwr_spec_all_trials)                
            ## get relative power for each freq and trial
            rpwr_all_trials = []
            for j, trial in enumerate(pwr_spec_all_trials):
                evoked_pwr_spec = pwr_spec_all_trials[j]
                ong_pwr_spec = ong_pwr_spec_all_trials[j]
                rpwr_spec = []
                for k, val in enumerate(evoked_pwr_spec):
                    rpwr_spec.append(10*math.log10(evoked_pwr_spec[k]/ong_pwr_spec[k]))
                rpwr_spec = pylab.array(rpwr_spec)
                rpwr_all_trials.append(rpwr_spec)
            rpwr = pylab.array(rpwr_all_trials)
            #rP_spec_all_trials = pwr_spec_all_trials/ong_pwr_spec_all_trials
            #rP_dict[epoch].append(numpy.mean(rP_spec_all_trials, axis = 0))
            rP_dict[epoch].append(numpy.mean(rpwr, axis = 0))

    #plot the rP spectra for each cell, and the across-cell average    
    max_all_epochs = 0
    for epoch in epochs:
        if epoch == 'transient':
            ax = ax1
            color = b
        else:
            ax = ax2
            color = g

        rP_all_cells = pylab.array(rP_dict[epoch])
        mean_rP = numpy.mean(rP_all_cells, axis = 0)
        std_rP = numpy.std(rP_all_cells, axis = 0)
        upper = mean_rP + std_rP
        lower_ = mean_rP - std_rP
        lower = lower_
        lower = []
        for val in lower_:
            if val < 0:
                lower.append(0)
            else:
                lower.append(val)
        max_for_epoch = max(mean_rP + upper)
        if max_for_epoch > max_all_epochs:
            max_all_epochs = max_for_epoch
        ax.plot(freqs, mean_rP, color = color, lw = 1.5)
        ax.plot(freqs, upper, color = color, lw = 1, alpha = 0.5)
        ax.plot(freqs, lower, color = color, lw = 1, alpha = 0.5)
        
        ax.fill_between(freqs, lower, upper, color = color, 
            alpha = 0.3)
        #ax.set_ylim(0, 400.0)
        
        #format the axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')                
    
#        locy = plticker.MultipleLocator(base = 200)
#        ax.yaxis.set_major_locator(locy)   
        locx = plticker.MultipleLocator(base = 20)
        ax.xaxis.set_major_locator(locx)

        ax.set_xlabel('frequency (Hz)', fontsize = size)
        if ax == ax1:
            y_labels = ax.get_yticks()
            ax.set_ylabel('rP (dB)', fontsize = size)
        else:
            y_labels = []
        #x_labels = ax.get_xticks()
        x_labels = ['', '', '', '40', '', '80', '']
        ax.set_xlim(0, 100)
        ax.set_yticklabels(y_labels, rotation = 'vertical',
            fontsize = size)
        ax.set_xticklabels(x_labels,
            fontsize = size)
        
     


