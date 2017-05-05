'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''
import os
import cPickle as pickle
from signal_processing.filtering import butterworth as butter
from signal_processing.replace_all_spikes import replace_all_spikes
from signal_processing.remove_sine_waves import remove_sine_waves as remove_sines
import pylab
import numpy   
import matplotlib.pyplot as plt

def get_processed_subtraces_for_cell(cell_name,
    windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0], 
    padding = 0.0, crit_freq = 100.0, filt_kind = 'low', replace_spikes = True, 
    remove_sine_waves = True, master_folder_path = 'E:\\correlated_variability'):
    
    '''
    For a given cell (indicated by cell_name), get an array containing
    subtraces of interest from all trials.  A subtrace starts (windows[0] + 
    gaps[0] + padding) ms before stim onset, and lasts for (sum(windows) + 
    sum(gaps) + 2*padding) ms.  Before pulling out the subtrace, filter and 
    possibly remove spikes from the full trace.  Remove sine waves from the 
    subtraces if desired.
    
    Parameters
    ---------- 
    cell_name : string
        name of cell of interest (e.g., '070314_c2')
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
    master_folder_path: string
        full path of directory containing data, code, figures, etc.

    Returns
    -------
    subtraces: numpy array
        array in which entries are subtraces extracted from each stimulus
        presentation (w/subtrace defined by stim onset, epochs, windows, gaps)
    
    '''

    raw_traces_path = master_folder_path + '\\downsampled_traces\\'
    if not os.path.isdir(raw_traces_path):
        os.path.mkdir(raw_traces_path)
    raw_traces_path += '%s\\'%cell_name    
    subtraces = []

    print 'processing traces for %s'%cell_name
    for trial in numpy.sort(os.listdir(raw_traces_path)):
        trial_path = raw_traces_path + trial
        results_dict = pickle.load(open(trial_path, 'rb'))
        
        trace = results_dict['voltage_traces_mV']
        samp_freq = results_dict['samp_freq']
        stim_onsets = results_dict['stim_onsets_ms']

        if replace_spikes == True:
            trace = replace_all_spikes(trace, samp_freq = samp_freq)   
        if filt_kind != 'unfiltered':
            trace = butter(trace, sampling_freq = samp_freq,
                critical_freq = crit_freq, kind = filt_kind,
                order = 3)

        for stim_onset in stim_onsets:
            start = int(int(stim_onset) - gaps[0] - windows[0] - padding)
            stop = int(int(stim_onset) + sum(gaps[1:]) + sum(windows[1:]) + padding)
            if start >= 0:
                sub_trace = trace[start:stop]              
                if remove_sine_waves:
                    #sub_trace = remove_sines(sub_trace, sine_freqs = [60])
                    sub_trace = remove_sines(sub_trace, 
                        sine_freqs = [60, 120, 180])
                subtraces.append(sub_trace)

    subtraces = pylab.array(subtraces)
    return subtraces
             

                    
                    
            