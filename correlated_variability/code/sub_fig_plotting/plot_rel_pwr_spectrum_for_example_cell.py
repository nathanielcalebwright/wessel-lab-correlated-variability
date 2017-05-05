'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

import matplotlib.ticker as plticker

from signal_processing.get_processed_subtraces_for_cell import \
get_processed_subtraces_for_cell as get_processed_traces

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

import nitime.algorithms as tsa
import pylab
import numpy
import scipy.io
import os
import cPickle as pickle
import math

def plot_rel_pwr_spectrum_for_example_cell(ax,
        fontproperties, cell_name = '070314_c2',
        windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
        padding = 0.0, hi_freq = 100.0, 
        replace_spikes = True, remove_sine_waves = True,
        get_new_processed_traces = False, 
        get_new_pwr_dict = False,
        master_folder_path = 'E:\\correlated_variability'):

    """
    For an example pair of simultaneously-recorded neurons, plot the 
    relative power spectrum (evoked/ongoing) for the transient and steady-
    state epochs.  Show the across-trial average spectra in high opacity,
    and the +/- 95% confidence intervals (via bootstrapping) in low opacity.
    
    Parameters
    ----------
    ax : matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    cell_name : string
        name of example cell to plot (e.g., '070314_c2')
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
    get_new_processed_traces: bool
        if True, start with unprocessed data (actually, downampled to 1 kHz
        from 10 kHz).  otherwise, use saved subtraces (numpy arrays) if they
        exist, to speed things up.
    get_new_pwr_dict: bool
        if True, get the subtraces, do power spectral analysis, etc.  
        otherwise, use saved results if they exist, to speed things up.
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    none
    """

    fonts = fontproperties
    size = fonts.get_size()
    epochs = ['ongoing', 'transient', 'steady-state']
    ong_start = int(padding)
    ong_stop = int(padding + windows[0])
    trans_start = int(ong_stop + gaps[0] + gaps[1])
    trans_stop = int(trans_start + windows[1])
    ss_start = int(trans_stop + gaps[2])
    ss_stop = int(ss_start + windows[2])
    starts = [ong_start, trans_start, ss_start]
    stops = [ong_stop, trans_stop, ss_stop]

    y = (1, 1, 0)
    b = (0.2, 0.6, 1)
    g = (0, 1, 0)
    colors = [y, b, g]
    
    ### check pwr_path for saved file, or save new pwrs to pwr_path ###        
    pwr_dict_path = master_folder_path + '\\saved_intermediate_results\\power_spectral_analysis\\'
    pwr_dict_path += 'pwr_spectra_%s_'%cell_name
    pwr_dict_path += str(windows) + '_ms_windows_'
    pwr_dict_path += str(gaps) + '_ms_gaps_'
    pwr_dict_path += str(padding) + '_ms_padding_'
    if replace_spikes:
        pwr_dict_path += '_spikes_removed'
    if remove_sine_waves:
        pwr_dict_path += '_sine_removed'
    pwr_dict_path += '.p'
    
    #first, check for saved power dict
    if os.path.isfile(pwr_dict_path) and get_new_pwr_dict == False:
        avg_pwr_dict = pickle.load(open(pwr_dict_path, 'rb'))

    else:        
        #get processed subtraces
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
        
        #see if these processed subtraces already exist
        if os.path.isfile(traces_path) and get_new_processed_traces == False:
            subtraces_for_cell = numpy.load(traces_path)
        #if not, get from raw data
        else:
            crit_freq = 100 #not used
            filt_kind = 'unfiltered'
            subtraces_for_cell = get_processed_traces(cell_name, windows,
                gaps, padding, crit_freq, filt_kind, replace_spikes,
                remove_sine_waves, master_folder_path)
            numpy.save(traces_path, subtraces_for_cell)                     
        
        #calculate power from subtraces  
        pwr_dict_all_trials = {}
        avg_pwr_dict = {}
        for epoch in epochs:
            pwr_dict_all_trials[epoch] = [] 
            avg_pwr_dict[epoch] = {'pwr':[], 'bootstraps':[]}
        avg_pwr_dict['freqs'] = []    
        
        ### get full-trial residuals, calculate power(t) for each band ###
        subtraces_for_cell = pylab.array(subtraces_for_cell)

        for subtrace in subtraces_for_cell:
            subtrace_resid = subtrace - numpy.mean(subtraces_for_cell,
                axis = 0)
            for j, epoch in enumerate(epochs):
                start = starts[j]
                stop = stops[j]
                epoch_trace = subtrace_resid[start:stop]
                freqs, psd, var_or_nu = tsa.spectral.multi_taper_psd(epoch_trace - numpy.mean(epoch_trace), 
                    Fs = 1000.0)
                pwr_dict_all_trials[epoch].append(psd[:int(hi_freq)])                
                avg_pwr_dict['freqs'] = freqs[:int(hi_freq)]
        avg_pwr_dict['pwr'] = pwr_dict_all_trials
                
        pickle.dump(avg_pwr_dict, open(pwr_dict_path, 'wb'))

    #plot results
    loc = plticker.MultipleLocator(base = 200)
    ax.set_ylabel('rP', fontsize = size)
    ax.set_xlabel('frequency (Hz)', fontsize = size)
    
    freqs = avg_pwr_dict['freqs']
    pwr_dict_all_trials = avg_pwr_dict['pwr']
    
    for n, epoch in enumerate(epochs[1:]):
        pwr = pylab.array(pwr_dict_all_trials[epoch])
        ong_pwr = pylab.array(pwr_dict_all_trials['ongoing'])

        rpwr = pwr/ong_pwr
        bootstraps = []
        for d, hi_freq in enumerate(rpwr[0]):
            rpwr_for_this_freq_all_trials = []
            for trial in rpwr:
                rpwr_for_this_freq_all_trials.append(trial[d])
            bootstrap = get_bootstrap(rpwr_for_this_freq_all_trials, 
                num_reps = 1000, conf_level = 0.95, num_comparisons = 1.0)
            bootstraps.append(bootstrap)
        mean_rpwr = numpy.mean(rpwr, axis = 0)
    
        upper = []
        lower = []
        for c, val in enumerate(mean_rpwr):
            bootstrap = bootstraps[c]
            lower_val = val - bootstrap[0]
            lower_val = bootstrap[0]
            if lower_val < 0:
                lower_val = 0
            upper_val = val + bootstrap[1]
            upper_val = bootstrap[1]
            lower.append(lower_val)
            upper.append(upper_val)
        if epoch == 'steady-state':
            label = 'steady-\nstate'
        else:
            label = epoch
        ax.plot(freqs, mean_rpwr, color = colors[1:][n], lw = 2, 
            label = label)
        ax.plot(freqs, upper, color = colors[1:][n], lw = 1, alpha = 0.5)
        ax.plot(freqs, lower, color = colors[1:][n], lw = 1, alpha = 0.5)
        
        ax.fill_between(freqs, lower, upper, color = colors[1:][n], 
            alpha = 0.3)
        ax.set_xlim(freqs[-1], freqs[0])
    

    ax.yaxis.set_major_locator(loc)    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')                
    ax.legend(frameon = False, fontsize = size, loc = 1)

    y_labels = ax.get_yticks()
    x_labels = ax.get_xticks()
    #x_labels = ['0', '20', '40', '60', '80', '100']
    ax.set_xlim(0, 100)
    ax.set_yticklabels(y_labels, rotation = 'vertical',
        fontsize = size)
    ax.set_xticklabels(x_labels,
        fontsize = size)
    ax.tick_params('both', length = 3, width = 1.0, which = 'major',
        direction = 'outward')

    #ax.set_xticklabels(x_labels, fontsize = size)


            

            



