import numpy
import os

from signal_processing.get_processed_subtraces_for_cell import \
get_processed_subtraces_for_cell as get_traces

def plot_example_traces_with_low_freq_FFT(ax1, ax2, fontproperties, 
        example_cells_to_plot, windows_for_FFT = [8000.0, 1000.0, 1000.0],
        windows_for_CC = [1000.0, 1000.0, 1000.0],
        gaps_for_FFT = [1000.0, 200.0, 200.0], 
        gaps_for_CC = [0.0, 200.0, 200.0], padding = 0.0,
        crit_freq_for_traces = 100.0, filt_kind_for_traces = 'low',
        freq_range_for_FFT = (1.0, 5.0),
        replace_spikes = True, remove_sine_waves = True,
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
        axis corresponding to cell in A
    ax2: matplotlib axis
        axis corresponding to cell in B
    fontproperties: FontProperties object
        dictates size and font of text
    example_cells_to_plot : python list of strings
        list of cells to plot (e.g., ['030415_c1', '070314_c5'])
    windows_for_FFT: list of floats
        widths of ongoing, transient, and steady-state windows (ms) for the
        full traces to be plotted.  The FFT will be calulated for THIS
        ongoing window.
    gaps_for_FFT: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms), for the full traces to be plotted.  The FFT will
        be calculated according to THESE gaps.
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        subtrace
    crit_freq_for_trace: float, or tuple of floats
        critical frequency for broad-band filtering of traces (e.g., 100.0).
        Only applies to plotted traces.  The FFT is calculated for unfiltered
        traces.
    filt_kind_for_traces: string
        type of filter to be used for traces (e.g., 'low' for lowpass).
        Only applies to plotted traces.  The FFT is calculated for unfiltered
        traces.
    freq_range_for_FFT: tuple of floats
        range of frequencies over which to sum Fourier coefficients (e.g., 
        (1.0, 5.0), as in Sachidhanandam, et al. 2013)
    replace_spikes: bool
        if True, detect spikes in membrane potential recordings, and replace
        via interpolation before filtering
    remove_sine_waves: bool
        if True, use sine-wave-detection algorithm to remove 60 Hz line noise
        from membrane potential recordings after removing spikes and filtering
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
    
    #background colors for each window of activity
    y = (1, 1, 0)
    b = (0.2, 0.6, 1)
    g = (0, 1, 0)
    
    subtraces = {}
    
    for cell_name in example_cells_to_plot:
        subtraces[cell_name] = []

        ### see if processed subtraces exist ###
        traces_path = master_folder_path + '\\processed_sub_traces\\'
        traces_path += cell_name + '_'       
        traces_path += str(windows_for_FFT) + '_ms_windows_'
        traces_path += str(gaps_for_FFT) + '_ms_gaps_'
        traces_path += str(padding) + '_ms_padding_'
        traces_path += '100.0_Hz_lowpass_filt'
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
                windows_for_FFT, gaps_for_FFT, padding, crit_freq_for_traces, 
                filt_kind_for_traces, 
                replace_spikes, remove_sine_waves, master_folder_path)
            numpy.save(traces_path, subtraces_for_cell)
        subtraces[cell_name] = subtraces_for_cell
    
    #now plot traces and calculate FFTs, using only the smaller number of
    #trials available
    num_trials = min(len(subtraces[example_cells_to_plot[0]]), 
        len(subtraces[example_cells_to_plot[1]]))
    
    for j, cell_name in enumerate(subtraces.keys()):
        if j == 0:
            ax = ax1
        else:
            ax = ax2
        subtraces_for_cell = subtraces[cell_name]
        
        #now plot each trial, calculate low-freq FFT for pre-stim window  
        low_FFTs = []
        for subtrace in subtraces_for_cell[:num_trials]:
            #align traces for clarity
            subtrace += -1*numpy.percentile(subtrace, 5)
            ax.plot(subtrace, color = 'k', alpha = 0.3, lw = 0.6)
            
            #pull out pre-stim window, calculate FFT
            pre_stim_trace = subtrace[:int(windows_for_FFT[0])]
            pre_stim_trace = pre_stim_trace - numpy.mean(pre_stim_trace)
            FFT = numpy.abs(numpy.fft.fft(pre_stim_trace))
            freqs = numpy.fft.fftfreq(pre_stim_trace.size, 0.001)   
            idx = numpy.argsort(freqs)
    
            low_FFT = []
            for j, val in enumerate(freqs[idx]):
                if freq_range_for_FFT[0] <= val <= freq_range_for_FFT[1]:
                    low_FFT.append(FFT[idx][j])
                    
            total_low_FFT = sum(low_FFT)/float(len(low_FFT))
            low_FFTs.append(total_low_FFT)
        
        avg_low_FFT = numpy.mean(low_FFTs)
    
        #add scale bars
        y_val = 20
        V_bar_xs = [2500, 2500]
        V_bar_ys = [y_val, y_val + 5]
        
        t_bar_xs = [2500, 3500]
        t_bar_ys = [y_val, y_val]
        
        ax.plot(V_bar_xs, V_bar_ys, 'k', lw = 1.5)
        ax.plot(t_bar_xs, t_bar_ys, 'k', lw = 1.5)
        
        ax.text(numpy.mean(t_bar_xs), y_val - 2, '1 s', fontsize = size,
            horizontalalignment = 'center')
    
        ax.text(t_bar_xs[0] - 200, numpy.mean(V_bar_ys), '5 mV', fontsize = size,
            horizontalalignment = 'right', verticalalignment = 'center')

        #indicate epoch windows with color        
        stim_onset = windows_for_FFT[0] + gaps_for_FFT[0]
        ong_start = stim_onset - windows_for_CC[0] - gaps_for_CC[0]
        ong_stop = ong_start + windows_for_CC[0]
        trans_start = ong_stop + gaps_for_CC[1]
        trans_stop = trans_start + windows_for_CC[1]
        ss_start = trans_stop + gaps_for_CC[2]
        ss_stop = ss_start + windows_for_CC[2]
        
        ax.axvspan(ong_start, ong_stop, ymin = 0.05, ymax = 1, 
            facecolor = y, ec = 'none', alpha = 0.3)
        
        ax.axvspan(trans_start, trans_stop, 
            ymin = 0.05, ymax = 1, facecolor = b, ec = 'none', alpha = 0.3)
        
        ax.axvspan(ss_start, ss_stop, 
            ymin = 0.05, ymax = 1, facecolor = g, ec = 'none', alpha = 0.3)
        
        #draw two-sided arrow indicated window used to calculate FFT
        arrow_start = 0.5*windows_for_FFT[0]
        arrow_dx = 0.5*windows_for_FFT[0] - 300
        arrow_y = -1.5
        ax.arrow(arrow_start, arrow_y, dx = arrow_dx, dy = 0, head_width = 1.5, 
            head_length = 200, overhang = 0.5, fc = 'r', ec = 'r')
        ax.arrow(arrow_start, arrow_y, dx = -1*arrow_dx, dy = 0, head_width = 1.5, 
            head_length = 200, overhang = 0.5, fc = 'r', ec = 'r')
        ax.vlines(arrow_start + arrow_dx + 250, arrow_y - 1, arrow_y + 1,
            color = 'r', lw = 1.0)
        ax.vlines(arrow_start - arrow_dx - 250, arrow_y - 1, arrow_y + 1,
            color = 'r', lw = 1.0)

        #print <FFT_delta> on plot
        FFT_label = r'$\langle FFT_\delta\rangle$ = ' + str(round(avg_low_FFT, 1)) + ' mV'
        label_x = arrow_start
        label_y = arrow_y - 2
        ax.text(label_x, label_y, FFT_label, fontsize = size,
            horizontalalignment = 'center')
        
        #format the axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    ax2.get_shared_y_axes().join(ax1, ax2)



