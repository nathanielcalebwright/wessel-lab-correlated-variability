import numpy
from scipy.stats import wilcoxon as wlcx
from scipy.stats import ttest_1samp
import os
import cPickle as pickle
import matplotlib.ticker as plticker

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

from analysis.get_CC_all_pairs import get_CC_all_pairs


y = (1, 1, 0)
g = (0, 1, 0)
b = (0.2, 0.6, 1)


def plot_trajectory(ax, fontproperties, CC_dict, bootstrap_dict, crit_freq):
    
    fonts = fontproperties
    size = fonts.get_size()
    markersize = 8
    
    
    if type(crit_freq) == tuple:
        band_name = str(crit_freq[0]) + '-' + str(crit_freq[1]) + ' Hz'
    else:
        band_name = '0.1 - ' + str(crit_freq) + ' Hz'

    
    ong_CCs = CC_dict['ongoing']
    ong_boots = bootstrap_dict['ongoing']
    trans_CCs = CC_dict['transient']
    trans_boots = bootstrap_dict['transient']
    ss_CCs = CC_dict['steady-state']
    ss_boots = bootstrap_dict['steady-state']
    #need left-right "jitter" to make overlapping dots distinguishable
    jitters = numpy.zeros(len(ong_CCs))
    for n, val in enumerate(jitters):
        jitters[n] += numpy.random.uniform(-0.05, 0.05)
    
    ### Get p-vals for stim modulation of population, draw on plot.
    
    ong_trans_p_val = wlcx(ong_CCs, trans_CCs)[1]
    trans_ss_p_val = wlcx(trans_CCs, ss_CCs)[1]
    ong_ss_p_val = wlcx(ong_CCs, ss_CCs)[1]
    ong_trans_stat = wlcx(ong_CCs, trans_CCs)[0]
    trans_ss_stat = wlcx(trans_CCs, ss_CCs)[0]
    ong_ss_stat = wlcx(ong_CCs, ss_CCs)[0]

    print '      <CC> ong = %s'%str(numpy.mean(ong_CCs)), '+/-', numpy.std(ong_CCs), '(P = %s; one-sided t-test)'%str(0.5*ttest_1samp(ong_CCs, 0)[1])
    print '      <CC> trans = %s'%str(numpy.mean(trans_CCs)), '+/-', numpy.std(trans_CCs), '(P = %s; one-sided t-test)'%str(0.5*ttest_1samp(trans_CCs, 0)[1])
    print '      <CC> ss = %s'%str(numpy.mean(ss_CCs)), '+/-', numpy.std(ss_CCs), '(P = %s; one-sided t-test)'%str(0.5*ttest_1samp(ss_CCs, 0)[1])
    
    print '      ong --> trans P = %s'%str(ong_trans_p_val), '(stat = %s)'%str(ong_trans_stat)
    print '      trans --> ss P = %s'%str(trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)
    print '      ong --> ss P = %s'%str(ong_ss_p_val), '(stat = %s)'%str(ong_ss_stat)  

    print '      corrected ong --> trans P = %s'%str(3*ong_trans_p_val), '(stat = %s)'%str(ong_trans_stat)
    print '      corrected trans --> ss P = %s'%str(3*trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)
    print '      corrected ong --> ss P = %s'%str(3*ong_ss_p_val), '(stat = %s)'%str(ong_ss_stat) 
    
    if band_name == '20-100Hz':    
        y_offset = 0.004
        y1 = 0.25
        y2 = 0.28
        y_min = -0.1
        y_max = 0.3001
        loc = plticker.MultipleLocator(base = 0.1)
    else:
        y_offset = 0.04
        y1 = 0.60
        y2 = 0.70
        y_min = -0.25
        y_max = 0.75
        loc = plticker.MultipleLocator(base = 0.25)
    
    y_offset = 0.004
    #y1 = max(max(ong_CCs), max(trans_CCs), max(ss_CCs)) - 0.025
    y1 = 0.24
    y2 = y1 + 0.05


    ### draw one line connecting each pair of epochs for the population ###
    ### line alpha and asterisks indicate Wilcoxon signed-rank p-val ###
    
    if ong_trans_p_val < 0.05/3.:
        if 0.001/3. <= ong_trans_p_val < 0.01/3.: #corrected for multiple comparisons
            ong_trans_p_for_plot = '**'
        elif ong_trans_p_val < 0.001/3.:
            ong_trans_p_for_plot = '***'                
        else:
            ong_trans_p_for_plot = '*'
        ax.plot([-0.1, 0.75], [y1, y1], color = 'k', lw = 2.0)

        
    else:
        x_pos = 0.325
        ong_trans_p_for_plot = ''
        ax.plot([-0.1, 0.75], [y1, y1], color = 'k', alpha = 0.35, lw = 2.0)

    x_pos = 0.325
    ax.text(x_pos, y1 + y_offset, ong_trans_p_for_plot, fontsize = 8, 
        horizontalalignment = 'center')
        
    if trans_ss_p_val < 0.05/3.:
        if 0.001/3. <= trans_ss_p_val < 0.01/3.:
            trans_ss_p_for_plot = '**'
        elif trans_ss_p_val < 0.001/3.:
            trans_ss_p_for_plot = '***'                
        else:
            trans_ss_p_for_plot = '*'

        ax.plot([1.3, 2.1], [y1, y1], color = 'k', lw = 2.0)

        
    else:
        trans_ss_p_for_plot = ''
        ax.plot([1.3, 2.1], [y1, y1], color = 'k', alpha = 0.35, lw = 2.0)

    x_pos = 1.7
    ax.text(x_pos, y1 + y_offset, trans_ss_p_for_plot, fontsize = 8, 
        horizontalalignment = 'center')
        
    if ong_ss_p_val < 0.05/3.:
        if 0.001/3. <= ong_ss_p_val < 0.01/3.:
            ong_ss_p_for_plot = '**'
        elif ong_ss_p_val < 0.001/3.:
            ong_ss_p_for_plot = '***'                
        else:
            ong_ss_p_for_plot = '*'
        ax.plot([-0.1, 2.1], [y2, y2], color = 'k', lw = 2.0)


    else:
        x_pos = 1.0
        ong_ss_p_for_plot = ''
        ax.plot([-0.1, 2.1], [y2, y2], color = 'k', alpha = 0.35, lw = 2.0)
    
    x_pos = 1.0
    ax.text(x_pos, y2 + y_offset, ong_ss_p_for_plot, fontsize = 8, 
        horizontalalignment = 'center')
        
    ### draw lines connecting markers across epochs for each pair,
    ### with line style determined by significance of change across
    ### epoch transition

    for i, ent in enumerate(trans_boots):
        trans_ong_overlap = False
        for entry in trans_boots[i]:
            if ong_boots[i][0] <= entry <= ong_boots[i][1]:
                trans_ong_overlap = True
                break
        for entry in ong_boots[i]:
            if trans_boots[i][0] <= entry <= trans_boots[i][1]:
                trans_ong_overlap = True
                break
        if trans_ong_overlap:
            ls = '-'
            alpha = 0.4
            lw = 0.7
        else:
            ls = '-'
            alpha = 1.0
            lw = 1.0
        ax.plot([0 + jitters[i], 1 + jitters[i]], [ong_CCs[i], trans_CCs[i]],
            color = 'k', ls = ls, lw = lw, alpha = alpha)
        
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
            lw = 0.7
        else:
            ls = '-'
            alpha = 1.0
            lw = 1.0
        ax.plot([1 + jitters[i], 2 + jitters[i]], [trans_CCs[i], ss_CCs[i]],
            color = 'k', ls = ls, lw = lw, alpha = alpha)

    ### Plot points for each pair (with fill based on significance relative
    ### to zero).

    #ongoing
    for q, entry in enumerate(ong_CCs):
        ong_zero_overlap = False
        if ong_boots[q][0] <= 0 <= ong_boots[q][1]:
            ong_zero_overlap = True
        if ong_zero_overlap:
            ong_mfc = (1, 1, 1)
            ong_alpha = 1.0
        else:
            ong_mfc = y
            ong_alpha = 1.0
        ax.plot(0 + jitters[q], ong_CCs[q], 'o', mec = 'k',
            mfc = ong_mfc, mew = 0.5, alpha = ong_alpha, 
            markersize = markersize, clip_on = False)

    #transient
    for c, val in enumerate(trans_CCs):
        trans_zero_overlap = False
        if trans_boots[c][0] <= 0 <= trans_boots[c][1]:
            trans_zero_overlap = True
        if trans_zero_overlap:
            trans_mfc = (1, 1, 1)
            trans_alpha = 1.0
        else:
            trans_mfc = b
            trans_alpha = 1.0
        ax.plot(1 + jitters[c], trans_CCs[c], 'o', mec = 'k',
            mfc = trans_mfc, mew = 0.5, alpha = trans_alpha,
            markersize = markersize, clip_on = False)
        
    #steady-state
    for d, val in enumerate(ss_CCs):
        ss_zero_overlap = False
        if ss_boots[d][0] <= 0 <= ss_boots[d][1]:
            ss_zero_overlap = True
        if ss_zero_overlap:
            ss_mfc = (1, 1, 1)
            ss_alpha = 1.0
        else:
            ss_mfc = g
            ss_alpha = 1.0
        ax.plot(2 + jitters[d], ss_CCs[d], 'o', mec = 'k',
            mfc = ss_mfc, mew = 0.5, alpha = ss_alpha,
            markersize = markersize, clip_on = False)

    ### configure the plot ###
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.101, 0.301)
    loc = plticker.MultipleLocator(base = 0.1)
    ax.yaxis.set_major_locator(loc)

    ax.set_xticks([0, 1, 2])
    labels = ['ongoing','transient', 'steady-\nstate']
    ax.set_xticklabels(labels, size = size)

    ax.tick_params('both', length = 5.5, width = 1.3, which = 'major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticklabels(ax.get_yticks(), fontsize = size)


def plot_CC_trajectories(ax, fontproperties, 
    stim_types = ['extended_stim', 'flash'],
    windows = [1000.0, 1000.0, 1000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'band',
    replace_spikes = True, remove_sine_waves = True,
    get_new_CC_results_all_pairs = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):
    
    '''Make CC trajectories for V-V pairs.  WARNING: the axis labels are
    hard-coded here to give attractive axes for the default settings
    only.  Axis labels will not in general be correct for other settings.

    Calculate the trial-averaged Pearson correlation coefficient (CC) for each
    pair in the dataset, for the ongoing, transient, and steady-state epochs, 
    for the crit_freq freq band.  Plot the results in "trajectory" view.  For 
    each pair, the significance of the across-epoch change is assessed by
    bootstrapping.  For the population, the significance of the across-epoch
    change is assessed via the Wilcoxon signed-rank test.
    
    Parameters
    ----------
    ax: matplotlib axis
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

    fonts = fontproperties
    size = fonts.get_size()    
    epochs = ['ongoing', 'transient', 'steady-state']
                
    CC_dict = {}
    bootstrap_ranges = {}
    
    for epoch in epochs:
        CC_dict[epoch] = []
        bootstrap_ranges[epoch] = []               

    ### populate the dictionaries ###
    CC_dict_by_pair_all_trials = get_CC_all_pairs(stim_types,
        windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
        remove_sine_waves, get_new_CC_results_all_pairs,
        get_new_processed_traces, master_folder_path)
    
        
    #CC_dict_by_pair_all_trials = {cell_name:{'ongoing':[CC_trial1, 
    #    ..., CC_trialn], ..., 'steady-state':[CC_trial1,
    #    ..., CC_trialn]]}}

    
    for cell_name in CC_dict_by_pair_all_trials:
        CC_dict_for_cell = CC_dict_by_pair_all_trials[cell_name]

        for epoch in epochs:
            CCs_for_epoch = CC_dict_for_cell[epoch]
            avg_CC = numpy.mean(CCs_for_epoch)
            bootstrap = get_bootstrap(CCs_for_epoch, num_reps = 1000,
                conf_level = 0.95, num_comparisons = 1) 
            
            ## add trial-avg CC and bootstrap to CC_dict
            CC_dict[epoch].append(avg_CC)
            bootstrap_ranges[epoch].append(bootstrap)


    ### report number of cells, pairs, and preparations ###
    experiments = []
    cells = []
    for pair_name in CC_dict_by_pair_all_trials:
        expt_date = pair_name.split('_')[0]
        if expt_date not in experiments:
            experiments.append(expt_date)
        cell_names = pair_name.split('_')[1].split('-')
        for cell_name in cell_names:
            full_cell_name = expt_date + cell_name
            if full_cell_name not in cells:
                cells.append(full_cell_name)
    print '    %s cells;'%str(len(cells)), \
          '%s pairs;'%str(len(CC_dict_by_pair_all_trials.keys())), \
          '%s turtles'%str(len(experiments))
    
    if stim_types == ['extended_stim']:
        y_label = 'CC (%s -'%str(int(crit_freq[0])) + ' %s Hz)'%str(int(crit_freq[1]))
        ax.set_ylabel(y_label, fontsize = size)

    
    plot_trajectory(ax, fontproperties, CC_dict, bootstrap_ranges, 
        crit_freq)
        

