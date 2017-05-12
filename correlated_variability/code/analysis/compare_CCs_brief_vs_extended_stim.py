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
from scipy.stats import ranksums

from analysis.get_CC_all_pairs import get_CC_all_pairs



def compare_CCs_brief_vs_extended_stim(
    windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'band',
    replace_spikes = True, remove_sine_waves = True,
    get_new_CC_results_all_pairs = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):
    
    '''
    Calculate the trial-averaged Pearson correlation coefficient (CC) for each
    pair in the dataset, for the ongoing, transient, and steady-state epochs, 
    for the crit_freq freq band, for each stimulus type.  Compare the two
    populations of results (i.e., extended stim and brief stim results) using
    the Wilcoxon rank-sum test.
    
    Parameters
    ----------

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
        via inteCColation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    get_new_FFT_results_all_pairs: bool
        if True, get the subtraces, calculte CC for each trial, etc.  
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
    '''
   
    epochs = ['ongoing', 'transient', 'steady-state']
                
    CC_dict_ext = {}
    CC_dict_flash = {}
    
    for epoch in epochs:
        CC_dict_ext[epoch] = []
        CC_dict_flash[epoch] = []

    ### populate the dictionaries ###
    CC_dict_by_pair_all_trials_ext = get_CC_all_pairs(['extended_stim'],
        windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
        remove_sine_waves, get_new_CC_results_all_pairs,
        get_new_processed_traces, master_folder_path)

    CC_dict_by_pair_all_trials_flash = get_CC_all_pairs(['flash'],
        windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
        remove_sine_waves, get_new_CC_results_all_pairs,
        get_new_processed_traces, master_folder_path)    
        
    #CC_dict_by_pair_all_trials_ext = {pair_name:{'ongoing':[CC_trial1, 
    #    ..., CC_trialn], ..., 'steady-state':[CC_trial1,
    #    ..., CC_trialn]]}}

    
    for pair_name in CC_dict_by_pair_all_trials_ext:
        CC_dict_for_pair = CC_dict_by_pair_all_trials_ext[pair_name]

        for epoch in epochs:
            CCs_for_epoch = CC_dict_for_pair[epoch]
            avg_CC = numpy.mean(CCs_for_epoch)
            
            ## add trial-avg CC to CC_dict
            CC_dict_ext[epoch].append(avg_CC)

    for pair_name in CC_dict_by_pair_all_trials_flash:
        CC_dict_for_pair = CC_dict_by_pair_all_trials_flash[pair_name]

        for epoch in epochs:
            CCs_for_epoch = CC_dict_for_pair[epoch]
            avg_CC = numpy.mean(CCs_for_epoch)
            
            ## add trial-avg CC to CC_dict
            CC_dict_flash[epoch].append(avg_CC)
    
    ### compare CCs for each epoch ###
    print ''
    print '  ### comparison of CCs by stim type ###'
    for epoch in epochs:
        print '    %s'%epoch
        CCs_flash = CC_dict_flash[epoch]
        CCs_ext = CC_dict_ext[epoch]
        stat, p = ranksums(CCs_flash, CCs_ext)
        print '      %s'%epoch, 'P = %s'%str(p), '(stat = %s)'%str(stat)



    
        

