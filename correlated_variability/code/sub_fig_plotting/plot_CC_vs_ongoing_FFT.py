from scipy.stats import linregress
import numpy
import math
import pylab
from analysis.get_CC_all_pairs import get_CC_all_pairs as get_CC
import matplotlib.ticker as plticker
import matplotlib as mpl

from analysis.get_low_freq_FFT_all_cells import \
get_low_freq_FFT_all_cells as get_FFT

from statistical_analysis.singer_correlation_comparisons import dependent_corr


def plot_CC_vs_ongoing_FFT(ax1, ax2, ax3, fontproperties,
        stim_types = ['extended_stim', 'flash'],
        windows_for_FFT = [8000.0, 2000.0, 1000.0], 
        windows_for_CC = [1000.0, 1000.0, 1000.0], 
        gaps_for_FFT = [1000.0, 200.0, 200.0],
        gaps_for_CC = [0.0, 200.0, 200.0],
        padding = 0.0,
        crit_freq_for_CC = (20.0, 100.0), filt_kind_for_CC = 'band',
        freq_range_for_FFT = (1.0, 5.0),
        replace_spikes = True, remove_sine_waves = True,
        get_new_CC_results_all_pairs = False,
        get_new_FFT_results_all_cells = False,
        get_new_processed_traces = False,
        master_folder_path = 'E:\\correlated_variability'):
    
    """
    For an example pair of neurons (not necessarily simultaneously recorded), 
    plot (in low opacity) responses to multiple presentations of a visual 
    stimulus.  Process the traces, if not already done (replace spikes, 
    remove sine waves, and filter (according to crit_freq and filt_kind)).  
    For each trial, calculate the sum of low-frequency Fourier coefficients
    for pre-stimulus activity.  Indicate the across-trial average on the plot.
    Note: plot formatting is based on the default filter settings, and for the 
    pair and trials displayed in the manuscript figure.
    
    Parameters
    ----------
    ax1: matplotlib axis
    ax2: matplotlib axis
    ax3: matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    stim_types : python list of strings
        include all cells corresponding to these stimulus types (e.g., 
        ['extended_stim', 'flash'])
    windows_for_FFT: list of floats
        widths of ongoing, transient, and steady-state windows (ms) for
        calculating the FFT.  The FFT note that only the result for the ongoing
        window will be used.
    windows_for_CC: list of floats
        widths of ongoing, transient, and steady-state windows (ms) for
        calculating CC.  
    gaps_for_FFT: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms), for calculating the FFT.
    gaps_for_CC: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms), for calculating CC.
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace
    crit_freq_for_CC: float, or tuple of floats
        critical frequency for filtering of traces (e.g., (20.0, 100.0)), for
        calculating CC.
    filt_kind_for_CC: string
        type of filter to be used for traces (e.g., 'band' for bandpass), for
        calculating CC.
    freq_range_for_FFT: tuple of floats
        range of frequencies over which to sum Fourier coefficients (e.g., 
        (1.0, 5.0), as in Sachidhanandam, et al. 2013)
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering.  Only used for calculating CC.
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering.
        Only used for calculating CC.
    get_new_CC_results_all_pairs: bool
        if True, calculate CC for processed sub-traces.  otherwise, use saved 
        results (python dictionaries) if they exist, to speed things up.    
    get_new_FFT_results_all_cells: bool
        if True, calculate ongoing FFT for unfiltered sub-traces.  otherwise, 
        use saved results (python dictionaries) if they exist, to speed things 
        up.
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
    mpl.rcParams['mathtext.default'] = 'regular'


    y = (1, 1, 0)
    b = (0.2, 0.6, 1)
    g = (0, 1, 0)
    colors = [y, b, g]

    CC_dict_all_pairs = get_CC(stim_types,
        windows = windows_for_CC, gaps = gaps_for_CC,
        padding = padding, crit_freq = crit_freq_for_CC, 
        filt_kind = filt_kind_for_CC,
        replace_spikes = replace_spikes, 
        remove_sine_waves = remove_sine_waves,
        get_new_CC_results_all_pairs = get_new_CC_results_all_pairs,
        get_new_processed_traces = get_new_processed_traces, 
        master_folder_path = master_folder_path)
    
    FFT_dict_all_cells = get_FFT(stim_types, 
        windows = windows_for_FFT, gaps = gaps_for_FFT, padding = padding,
        freq_range_for_FFT = freq_range_for_FFT, 
        replace_spikes = replace_spikes, 
        remove_sine_waves = remove_sine_waves,
        get_new_FFT_results_all_cells = get_new_FFT_results_all_cells, 
        get_new_processed_traces = get_new_processed_traces,
        master_folder_path = master_folder_path)
    
    epochs = ['ongoing', 'transient', 'steady-state']    
    avg_CCs_all_pairs = {}
    for epoch in epochs:
        avg_CCs_all_pairs[epoch] = []
    max_FFT_all_pairs = []
    
    for pair_name in CC_dict_all_pairs:
        cell1_name = pair_name.split('-')[0]
        cell2_name = pair_name.split('-')[1]

        min_len = min(len(FFT_dict_all_cells[cell1_name]['ongoing']), 
            FFT_dict_all_cells[cell2_name]['ongoing'])

        low_FFT1 = numpy.mean(FFT_dict_all_cells[cell1_name]['ongoing'][:min_len])
        low_FFT2 = numpy.mean(FFT_dict_all_cells[cell2_name]['ongoing'][:min_len])

        use_data = False

        if len(CC_dict_all_pairs[pair_name]) > 0 and len(FFT_dict_all_cells[cell1_name]['ongoing']) > 0 and len(FFT_dict_all_cells[cell2_name]['ongoing']) > 0:
            use_data = True
        if use_data == True:
            for epoch in epochs:
                avg_CCs_all_pairs[epoch].append(numpy.mean(CC_dict_all_pairs[pair_name][epoch][:min_len]))
    
            max_FFT = max(low_FFT1, low_FFT2)
            max_FFT_all_pairs.append(max_FFT)
    
    r_vals = []
    print ''
    print '### CC vs. ongoing <FFT_delta>' ###
    for k, epoch in enumerate(epochs):
        if k == 0:
            ax = ax1
            locy = plticker.MultipleLocator(base = 0.04)   
            fit_x = 0.3
            fit_y = 0.527
        elif k == 1:
            ax = ax2
            locy = plticker.MultipleLocator(base = 0.05)
            fit_x = 0.3
            fit_y = 0.8
        else:
            ax = ax3
            locy = plticker.MultipleLocator(base = 0.04)
            x_label = r'$max(\langle FFT_{\delta,1}\rangle , \langle FFT_{\delta,2}\rangle$) (mV)'

            ax.set_xlabel(x_label, fontsize = size)
            fit_x = 0.3
            fit_y = 0.8

        avg_CCs_all_pairs[epoch] = pylab.array(avg_CCs_all_pairs[epoch])
        slope, intercept, r, p, std_err = linregress(max_FFT_all_pairs, 
            abs(avg_CCs_all_pairs[epoch]))
        r_vals.append(r)

        print '  %s'%epoch        
        print '    r =', r, '(p = %s)'%str(p)
        ax.plot(max_FFT_all_pairs, abs(avg_CCs_all_pairs[epoch]), 'o',
            color = colors[k], markersize = 4, mec = 'k', mew = 0.6, 
            alpha = 0.5, clip_on = False)

        #print fit results on axis
        r_string = str(round(r, 2))
        p_string = str(round(p, 3))
        
        fit_string = 'r = %s'%r_string + '\np = %s'%p_string
        ax.text(fit_x, fit_y, fit_string, fontsize = size, 
            horizontalalignment = 'center', 
            verticalalignment = 'bottom',
            transform = ax.transAxes)

        fit_ys = []
        fit_xs = []
        x = min(max_FFT_all_pairs)
        while x <= max(max_FFT_all_pairs):
            fit_xs.append(x)
            y = slope*x + intercept
            fit_ys.append(y)
            x += 10

        ax.plot(fit_xs, fit_ys, color = 'r', lw = 1.5)

        y_label = '|CC| (%s -'%str(int(crit_freq_for_CC[0])) + ' %s Hz)'%str(int(crit_freq_for_CC[1])) + '\n%s'%epoch
        ax.set_ylabel(y_label, fontsize = size)
    
        #format the axis
        ax.tick_params('both', length = 3, width = 1.0, which = 'major',
            direction = 'outward')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 

        locx = plticker.MultipleLocator(base = 200.0)
        ax.xaxis.set_major_locator(locx)
        ax.yaxis.set_major_locator(locy)

        x_labels = ax.get_xticks()
        y_labels = ax.get_yticks()
        x_labels = ax.get_xticks()
        ax.set_yticklabels(y_labels, fontsize = size)
        ax.set_xticklabels(x_labels, fontsize = size)


    #get p-val for comparison of three r-values
    t_comp, p_comp = dependent_corr(r_vals[0], r_vals[1], r_vals[2], 
        len(avg_CCs_all_pairs['ongoing']))

    print ''    
    print '### comparison of r-values ###'
    print '    t = %s'%str(t_comp)
    print '    p = %s'%str(p_comp)
