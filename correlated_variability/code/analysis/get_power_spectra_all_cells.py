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

import nitime.algorithms as tsa

import os
import cPickle as pickle
import numpy


def get_power_spectra_all_cells(stim_types = ['extended_stim', 'flash'],
    windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0),
    replace_spikes = True, remove_sine_waves = True,
    get_new_pwr_results_all_cells = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):    
    
    '''
    For each cell of interest (defined by stim_types selection), get the
    power spectrum for each trial and epoch.
    
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
    pwr_dict_all_cells: python dictionary
        dictionary of power spectra for all trials and epochs, for each cell
        of interest (defined by stim_types).
    
    '''

    epochs = ['ongoing', 'transient', 'steady-state'] 
    ong_start = int(padding)
    ong_stop = int(padding + windows[0])
    trans_start = int(ong_stop + gaps[0] + gaps[1])
    trans_stop = int(trans_start + windows[1])
    ss_start = int(trans_stop + gaps[2])
    ss_stop = int(ss_start + windows[2])
    starts = [ong_start, trans_start, ss_start]
    stops = [ong_stop, trans_stop, ss_stop]

    ### check pwr_path for saved file, or save new pwrs to pwr_path ###        
    pwr_dict_path = master_folder_path + '\\saved_intermediate_results\\power_spectral_analysis\\'
    if not os.path.isdir(pwr_dict_path):
        os.mkdir(pwr_dict_path)
    pwr_dict_path += 'pwr_spectra_all_cells_and_trials_'
    for stim_type in stim_types:                
        pwr_dict_path += '%s_'%stim_type
    pwr_dict_path += str(windows) + '_ms_windows_'
    pwr_dict_path += str(gaps) + '_ms_gaps_'
    pwr_dict_path += str(padding) + '_ms_padding'
    if replace_spikes:
        pwr_dict_path += '_spikes_removed'
    if remove_sine_waves:
        pwr_dict_path += '_sine_removed'
    pwr_dict_path += '.p'
    
    ### use saved pwr dictionary, if it exists ###
    if os.path.isfile(pwr_dict_path) and get_new_pwr_results_all_cells == False:
        pwr_dict_all_cells = pickle.load(open(pwr_dict_path, 'rb'))

    ### or make a new pwr dictionary ###
    else:
        print 'getting new power spectrum dict'
        pwr_dict_all_cells = {}
        
        ### get dictionary of all cells of interest (segregated by stim_type) ###
        cells_by_stim_type = get_cells_by_stim_type(stim_types, 
            master_folder_path)
        for stim_type in stim_types:
            cells_for_stim_type = cells_by_stim_type[stim_type]
            for cell_name in cells_for_stim_type:
                pwr_dict_all_cells[cell_name] = {}
                for epoch in epochs:
                    pwr_dict_all_cells[cell_name][epoch] = {'pwr':[], 
                        'freqs':[]}
                    
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
                
                #now get pwr spectrum for each epoch and trial for this cell
                print 'getting power spectrum for %s'%cell_name
                for subtrace in subtraces_for_cell:
                    subtrace_resid = subtrace - numpy.mean(subtraces_for_cell,
                        axis = 0)
                    for j, epoch in enumerate(epochs):
                        start = starts[j]
                        stop = stops[j]
                        epoch_trace = subtrace_resid[start:stop]
                        freqs, psd, var_or_nu = tsa.spectral.multi_taper_psd(epoch_trace - numpy.mean(epoch_trace), 
                            Fs = 1000.0)
                        pwr_dict_all_cells[cell_name][epoch]['freqs'].append(freqs)
                        pwr_dict_all_cells[cell_name][epoch]['pwr'].append(psd)

        
        pickle.dump(pwr_dict_all_cells, open(pwr_dict_path, 'wb'))

    return pwr_dict_all_cells

