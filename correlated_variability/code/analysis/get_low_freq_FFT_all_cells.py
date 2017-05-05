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

from general.get_cells_by_stim_type import get_cells_by_stim_type

import os
import cPickle as pickle
import numpy
import matplotlib.pyplot as plt

def get_low_freq_FFT_all_cells(stim_types = ['extended_stim', 'flash'],
    windows = [9000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, freq_range_for_FFT = (1.0, 5.0),
    replace_spikes = True, remove_sine_waves = False,
    get_new_FFT_results_all_cells = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):  
    
    
    '''
    For each cell of interest (defined by stim_types selection), get the
    total low-freq (1 - 5 Hz) FFT of the membrane potential (as in 
    as in Sachidhanandam et al. 2013) for each trial and epoch.
    
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
    freq_range_for_FFT: float, or tuple of floats
        frequency range over which to average (e.g., (20.0, 100.0))
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
    get_new_FFT_results_all_cells: bool
        if True, get the subtraces, calculte <FFT_delta> for each trial, etc.  
        otherwise, use saved results if they exist, to speed things up.
    get_new_processed_traces: bool
        if True, start with unprocessed data (actually, downampled to 1 kHz
        from 10 kHz).  otherwise, use saved subtraces (numpy arrays) if they
        exist, to speed things up.
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    FFT_dict_all_cells: python dictionary
        dictionary of <FFT_delta> for all trials and epochs, for each cell
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
    FFT_dict_path = master_folder_path + '\\saved_intermediate_results'
    FFT_dict_path += '\\power_spectral_analysis\\'
    if not os.path.isdir(FFT_dict_path):
        os.mkdir(FFT_dict_path)
    FFT_dict_path += 'FFT_all_cells_and_trials_'
    for stim_type in stim_types:                
        FFT_dict_path += '%s_'%stim_type
    FFT_dict_path += str(windows) + '_ms_windows_'
    FFT_dict_path += str(gaps) + '_ms_gaps_'
    FFT_dict_path += str(padding) + '_ms_padding_'
    FFT_dict_path += str(freq_range_for_FFT) + '_Hz_range'
    if replace_spikes:
        FFT_dict_path += '_spikes_removed'
    if remove_sine_waves:
        FFT_dict_path += '_sine_removed'
    FFT_dict_path += '.p'
    

    ### use saved pwr dictionary, if it exists ###
    if os.path.isfile(FFT_dict_path) and get_new_FFT_results_all_cells == False:
        FFT_dict_all_cells = pickle.load(open(FFT_dict_path, 'rb'))

    ### or make a new pwr dictionary ###
    else:
        FFT_dict_all_cells = {}
        
        ### get dictionary of all cells of interest (segregated by stim_type) ###
        cells_by_stim_type = get_cells_by_stim_type(stim_types, 
            master_folder_path)
        for stim_type in stim_types:
            cells_for_stim_type = cells_by_stim_type[stim_type]
            for cell_name in cells_for_stim_type:
                FFT_dict_all_cells[cell_name] = {}
                for epoch in epochs:
                    FFT_dict_all_cells[cell_name][epoch] = []
                    
                ### see if processed subtraces exist ###                    
                traces_path = master_folder_path + '\\processed_sub_traces\\'
                traces_path += cell_name + '_'       
                traces_path += str(windows) + '_ms_windows_'
                traces_path += str(gaps) + '_ms_gaps_'
                traces_path += str(padding) + '_ms_padding_'
                traces_path += 'unfiltered'
                if replace_spikes:
                    traces_path += '_spikes_removed'
                if remove_sine_waves:
                    traces_path += '_sine_removed'
                traces_path += '.npy'                
                if os.path.isfile(traces_path) and get_new_processed_traces == False:
                    subtraces_for_cell = numpy.load(traces_path)
                #if not, get from raw data
                else:
                    subtraces_for_cell = get_traces(cell_name, 
                        windows, gaps, padding, crit_freq = 100.0, 
                        filt_kind = 'unfiltered', 
                        replace_spikes = replace_spikes, 
                        remove_sine_waves = remove_sine_waves, 
                        master_folder_path = master_folder_path)
                    numpy.save(traces_path, subtraces_for_cell)
                
                #now get <FFT_delta> for each epoch and trial for this cell
                
                for subtrace in subtraces_for_cell:
                    for j, epoch in enumerate(epochs):
                        start = sub_intervals[j][0]
                        end = sub_intervals[j][1]
                        epoch_trace = subtrace[start:end]
                        if len(epoch_trace) > 0:
                            epoch_trace = epoch_trace - numpy.mean(epoch_trace)
                            FFT = numpy.abs(numpy.fft.fft(epoch_trace))
                            freqs = numpy.fft.fftfreq(epoch_trace.size, 0.001)   
                            idx = numpy.argsort(freqs)
    
                            low_freqs = []
                            low_pwr = []
                            
                            for j, val in enumerate(freqs[idx]):
                                if 1 <= val <= 5:
                                    low_freqs.append(val)
                                    low_pwr.append(FFT[idx][j])
                                    
                            total_low_pwr = sum(low_pwr)/float(len(low_pwr))
                            FFT_dict_all_cells[cell_name][epoch].append(total_low_pwr) 

        
        pickle.dump(FFT_dict_all_cells, open(FFT_dict_path, 'wb'))

    return FFT_dict_all_cells

