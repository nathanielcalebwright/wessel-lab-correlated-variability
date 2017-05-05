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
get_processed_subtraces_for_cell as get_traces

from general.get_cell_pairs_by_stim_type import get_cell_pairs_by_stim_type

import os
import cPickle as pickle
import numpy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def get_CC_all_pairs(stim_types = ['extended_stim', 'flash'],
    windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'band',
    replace_spikes = True, remove_sine_waves = True,
    get_new_CC_results_all_pairs = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):    
    
    '''
    For each pair pair of interest (defined by stim_types selection), get the
    Pearson correlation coefficient of the membrane potentials for each trial 
    and epoch.
    
    Parameters
    ----------
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
    CC_dict_all_pairs: python dictionary
        dictionary of CC for all trials and epochs, for each pair
        of interest (defined by stim_types).
    '''
    #relevant indices for subtraces
    sub_ong_start = int(padding)
    sub_ong_stop = int(windows[0] + padding)
    sub_trans_start = int(windows[0] + gaps[0] + gaps[1] + padding)
    sub_trans_stop = int(sub_trans_start + windows[1])
    sub_ss_start = int(sub_trans_stop + gaps[2])
    sub_ss_stop = int(sub_ss_start + windows[2])
    sub_intervals = [[sub_ong_start, sub_ong_stop], 
        [sub_trans_start, sub_trans_stop], [sub_ss_start, sub_ss_stop]]
    
    epochs = ['ongoing', 'transient', 'steady-state']    

    ### check pwr_path for saved file, or save new pwrs to pwr_path ###        
    CC_dict_path = master_folder_path + '\\saved_intermediate_results\\'
    CC_dict_path += 'CC\\'
    if not os.path.isdir(CC_dict_path):
        os.mkdir(CC_dict_path)
    CC_dict_path += 'CC_all_pairs_and_trials_'
    for stim_type in stim_types:                
        CC_dict_path += '%s_'%stim_type
    CC_dict_path += str(windows) + '_ms_windows_'
    CC_dict_path += str(gaps) + '_ms_gaps_'
    CC_dict_path += str(padding) + '_ms_padding_'
    CC_dict_path += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
    if replace_spikes:
        CC_dict_path += '_spikes_removed'
    if remove_sine_waves:
        CC_dict_path += '_sine_removed'
    CC_dict_path += '.p'
    
    ### use saved pwr dictionary, if it exists ###
    if os.path.isfile(CC_dict_path) and get_new_CC_results_all_pairs == False:
        CC_dict_all_pairs = pickle.load(open(CC_dict_path, 'rb'))

    ### or make a new pwr dictionary ###
    else:
        print 'getting new CC_dict_all_pairs'
        CC_dict_all_pairs = {}
        
        ### get dictionary of all pairs of interest (segregated by stim_type) ###
        pairs_by_stim_type = get_cell_pairs_by_stim_type(stim_types, 
            master_folder_path)
        for stim_type in stim_types:
            pairs_for_stim_type = pairs_by_stim_type[stim_type]
            for pair in pairs_for_stim_type:
                cell_name1 = pair[0]
                cell_name2 = pair[1]   
                pair_name = cell_name1 + '-' + cell_name2
                CC_dict_all_pairs[pair_name] = {}
                for epoch in epochs:
                    CC_dict_all_pairs[pair_name][epoch] = []
                    
                ### see if processed subtraces exist ###
                traces_path1 = master_folder_path + '\\processed_sub_traces\\'
                traces_path1 += cell_name1 + '_'       
                traces_path1 += str(windows) + '_ms_windows_'
                traces_path1 += str(gaps) + '_ms_gaps_'
                traces_path1 += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
                if replace_spikes:
                    traces_path1 += '_spikes_removed'
                if remove_sine_waves:
                    traces_path1 += '_sine_removed'
                traces_path1 += '.npy'                
                if os.path.isfile(traces_path1) and get_new_processed_traces == False:
                    subtraces_cell1 = numpy.load(traces_path1)
                #if not, get from raw data
                else:
                    if type(crit_freq) == float:
                        filt_kind = 'low'
                    else:
                        filt_kind = 'band'
                    subtraces_cell1 = get_traces(cell_name1, 
                        windows, gaps, padding, crit_freq, filt_kind, 
                        replace_spikes, remove_sine_waves, master_folder_path)
                    numpy.save(traces_path1, subtraces_cell1)

                traces_path2 = master_folder_path + '\\processed_sub_traces\\'
                traces_path2 += cell_name2 + '_'       
                traces_path2 += str(windows) + '_ms_windows_'
                traces_path2 += str(gaps) + '_ms_gaps_'
                traces_path2 += str(padding) + '_ms_padding_'
                traces_path2 += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
                if replace_spikes:
                    traces_path2 += '_spikes_removed'
                if remove_sine_waves:
                    traces_path2 += '_sine_removed'
                traces_path2 += '.npy'                
                if os.path.isfile(traces_path2) and get_new_processed_traces == False:
                    subtraces_cell2 = numpy.load(traces_path2)
                #if not, get from raw data
                else:
                    if type(crit_freq) == float:
                        filt_kind = 'low'
                    else:
                        filt_kind = 'band'
                    subtraces_cell2 = get_traces(cell_name2, 
                        windows, gaps, padding, crit_freq, filt_kind, 
                        replace_spikes, remove_sine_waves, master_folder_path)
                    numpy.save(traces_path2, subtraces_cell2)
                
                #only use the number of trials in the shorter array (this
                #consideration necessary b/c some patches last longer than
                #others)
                if len(subtraces_cell1) >= len(subtraces_cell2):                    
                    max_len = len(subtraces_cell2)
                else:
                    max_len = len(subtraces_cell1)
                
                subtraces1_for_calc = subtraces_cell1[:max_len]
                subtraces2_for_calc = subtraces_cell2[:max_len]
                    
                for k, subtrace in enumerate(subtraces1_for_calc):
                    #get residual traces for each trial
                    resid1 = subtraces1_for_calc[k] - numpy.mean(subtraces1_for_calc,
                        axis = 0)
                    resid2 = subtraces2_for_calc[k] - numpy.mean(subtraces2_for_calc,
                        axis = 0)
#                    plt.figure()
#                    plt.plot(resid1, 'r')
#                    plt.plot(resid2, 'k')
#                    plt.title(pair_name)
                    
                    #calculate CC (or Pearson r) for each epoch
                    for j, epoch in enumerate(epochs):
                        #plt.figure()
                        start = sub_intervals[j][0]
                        end = sub_intervals[j][1]
                        epoch_resid1 = resid1[start:end]
                        epoch_resid2 = resid2[start:end]
#                        plt.plot(epoch_resid1, 'r')
#                        plt.plot(epoch_resid2, 'k')
#                        plt.title(epoch)
                        CC = pearsonr(epoch_resid1, epoch_resid2)[0]
                        CC_dict_all_pairs[pair_name][epoch].append(CC)
                    #plt.show()
        
        pickle.dump(CC_dict_all_pairs, open(CC_dict_path, 'wb'))

    return CC_dict_all_pairs

