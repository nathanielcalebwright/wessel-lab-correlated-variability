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
from scipy.stats import wilcoxon as wlcx
from scipy.stats import ttest_1samp
import os
import cPickle as pickle
import matplotlib.ticker as plticker

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

from analysis.get_CC_all_pairs_model_network import \
get_CC_all_pairs_model_network as get_CC_all_pairs


y = (1, 1, 0)
g = (0, 1, 0)
b = (0.2, 0.6, 1)


def plot_trajectory(ax, fontproperties, CC_dict, bootstrap_dict, crit_freq,
        model_version):
    
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

    print '    <CC> ong = %s'%str(numpy.mean(ong_CCs)), '+/-', numpy.std(ong_CCs)
    print '      (P = %s; one-sided t-test)'%str(0.5*ttest_1samp(ong_CCs, 0)[1])
    print '    <CC> trans = %s'%str(numpy.mean(trans_CCs)), '+/-', numpy.std(trans_CCs)
    print '      (P = %s; one-sided t-test)'%str(0.5*ttest_1samp(trans_CCs, 0)[1])  
    print '    <CC> ss = %s'%str(numpy.mean(ss_CCs)), '+/-', numpy.std(ss_CCs) 
    print '      (P = %s; one-sided t-test)'%str(0.5*ttest_1samp(ss_CCs, 0)[1])
    
    print '    ong --> trans P = %s'%str(ong_trans_p_val), '(stat = %s)'%str(ong_trans_stat)
    print '    trans --> ss P = %s'%str(trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)
    print '    ong --> ss P = %s'%str(ong_ss_p_val), '(stat = %s)'%str(ong_ss_stat)  

    print '    corrected ong --> trans P = %s'%str(3*ong_trans_p_val), '(stat = %s)'%str(ong_trans_stat)
    print '    corrected trans --> ss P = %s'%str(3*trans_ss_p_val), '(stat = %s)'%str(trans_ss_stat)
    print '    corrected ong --> ss P = %s'%str(3*ong_ss_p_val), '(stat = %s)'%str(ong_ss_stat) 
    
    y_offset = 0.004
    y1 = 0.42
    y2 = 0.48
    y_min = -0.101
    y_max = 0.501
    loc = plticker.MultipleLocator(base = 0.1)


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
    ax.set_ylim(y_min, y_max)
#    y_labels_ = ax.get_yticks()
#    y_labels = []
#
#    for h, label in enumerate(y_labels_):
#        if (h-1)%3 == 0:
#            y_labels.append(label)
#        else:
#            y_labels.append('')
#    ax.set_yticklabels(y_labels, fontsize = size)
    #labels = ['-0.1', '', '0.1', '', '0.3']
    ax.yaxis.set_major_locator(loc)
    #ax.set_yticklabels(labels, size = size)

    
    #ax.locator_params(axis = 'y', nbins = 4)

        
    #plt.ylabel('<CC(0)>', fontsize = 14)
    ax.set_xticks([0, 1, 2])
    labels = ['ongoing','transient', 'steady-\nstate']
    #labels = ['', '', '']
    ax.set_xticklabels(labels, size = size)
    ax.set_yticklabels(ax.get_yticks(), size = size)

    ax.tick_params('both', length = 5.5, width = 1.3, which = 'major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    y_label = 'CC ('
    if type(crit_freq) == float:
        y_label += str(int(crit_freq)) + ' Hz)'
    else:
        y_label += str(int(crit_freq[0])) + '-' + str(int(crit_freq[1])) + ' Hz)'
    ax.set_ylabel(y_label, fontsize = size)


def plot_CC_trajectories_model_neurons(ax, fontproperties,
        node_group = '1_center', N_E = 800, N_I = 200, tau_Em = 50, 
        tau_Im = 25, p_exc = 0.03, p_inh = 0.20, tau_ref_E = 10, tau_ref_I = 5,
        V_leak_E_min = -70, V_leak_I_min = -70, V_th = -40, V_reset = -59,
        V_peak = 0, V_syn_E = 50, V_syn_I = -68, C_E = 0.4, C_I = 0.2,
        g_AE = 3.0e-3, g_AI = 6.0e-3, g_GE = 30.0e-3, g_GI = 30.0e-3, 
        g_extE = 6.0e-3, g_extI = 6.0e-3, g_leak_E = 10.0e-3, g_leak_I = 5.0e-3,
        tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, tau_AId = 1.0, tau_AEl = 1.5, 
        tau_AEr = 0.2, tau_AEd = 1.0, 
        tauR_intra = 50000.0, tauD_intra = 300.0, tauR_extra = 50000.0, 
        tauD_extra = 300.0, rateI = 65, rateE = 65, stim_factor = 4.5, 
        num_nodes_to_save = 20, num_pairs = 40, num_LFPs = 8, 
        windows = [1000.0, 1000.0, 1000.0], 
        gaps = [200.0, 0.0, 200.0], padding = 500.0, 
        stim_type = 'linear_increase', time_to_max = 200, 
        exc_connectivity = 'clustered', inh_connectivity = 'random',
        exc_syn_weight_dist = 'beta', net_state = 'synchronous',
        ext_SD = True, int_SD = True, 
        generate_LFPs = True, save_results = True,
        num_trials = 15, crit_freq = (20.0, 100.0), filt_kind = 'band', 
        run_new_net_sim = False,
        get_new_CC_results_all_pairs = False,
        master_folder_path = 'F:\\correlated_variability'):

    
    '''Make CC trajectories for pairs of 'test' neurons.  'Test' neurons
    receive synaptic inputs generated by the model network.  WARNING: the axis 
    labels are hard-coded here to give attractive axes for the default settings
    only.  Axis labels will not in general be correct for other settings.

    Calculate the trial-averaged Pearson correlation coefficient (CC) for each
    test pair, for the ongoing, transient, and steady-state epochs, 
    for the crit_freq freq band.  Plot the results in "trajectory" view.  For 
    each pair, the significance of the across-epoch change is assessed by
    bootstrapping.  For the population, the significance of the across-epoch
    change is assessed via the Wilcoxon signed-rank test.
    
    
    Parameters
    ----------
    ax: matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    node_group: float
        group of network nodes from which to select example Vs
    tau_Em: float
        membrane time constant in ms for exc nodes
    tau_Im: float
        membrane time constant in ms for inh nodes
    N_E: float or int
        number of exc nodes
    N_I: float or int
        number of inh nodes
    p_exc: float
        connection probability for E --> E and E --> I
    p_inh: float
        connection probability for I --> E and I --> I
    tau_ref_E: float
        absolute refractory period for exc nodes in ms
    tau_ref_I: float
        absolute refractory period for inh nodes in ms
    V_leak_I: float or int
        leak reversal potential for inh nodes in mV
    V_leak_E: float or int
        leak reversal potential for exc nodes in mV
    V_th: float or int
        spike threshold for all nodes in mV
    V_reset: float or int
        post-spike reset membrane potential for all nodes in mV
    V_peak: float or int
        peak spike membrane potential for all nodes in mV
    dt: float
        time step in ms
    V_syn_E: float or int
        synaptic reversal potential for exc nodes in mV
    V_syn_I: float or int
        synaptic reversal potential for inh nodes in mV
    g_AE: float
        synaptic conductance for AMPA channels on exc nodes in microS
    g_AI: float 
        synaptic conductance for AMPA channels on inh nodes in microS
    g_GE: float 
        synaptic conductance for GABA channels on exc nodes in microS
    g_GI: float 
        synaptic conductance for GABA channels on inh nodes in microS
    tau_Fl: float or in f
        inh to inh delay time constant in ms
    tau_Fr: float
        inh to inh rise time constant in ms
    tau_Fd: float
        inh to inh decay time constant in ms
    tau_AIl: float
        exc to inh AMPA delay time constant in ms
    tau_AIr: float
        exc to inh AMPA rise time constant in ms
    tau_AId: float 
        exc to inh AMPA decay time constant in ms
    tau_AEl: float
        exc to exc AMPA delay time constant in ms
    tau_AEr: float
        exc to exc AMPA rise time constant in ms
    tau_AEd: float 
        exc to exc AMPA decay time constant in ms
    g_extE: float
        synaptic conductance for ext input on exc in microS
    g_extI: float
        synaptic conductance for ext input on inh in microS
    C_E: float 
        capacitance for exc nodes in nF
    C_I: float 
        capacitance for inh nodes in nF
    g_leak_E: float
        leak conductance for exc nodes in microS
    g_leak_I: float l
        leak conductance for inh nodes in microS
    rateI: float or int 
        ongoing external input rate for exc nodes in Hz
    rateE: float or int 
        ongoing external input rate for inh nodes in Hz
    stim_factor: float or int
        factor by which external input increases after stim onset 
    num_nodes_to_save: int
        number of nodes for which to save synaptic inputs (for injecting
        into 'test' neurons in a separate function) 
    num_pairs: int
        number of test neuron pairs for which to calculate CCs
    num_LFPs: float
        number of LFPs to simulate (determines # of nodes in each 'electrode')
    windows: list of floats
        widths of ongoing, transient, and steady-state windows (ms)
    gaps: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms)
    padding: float
        size of window (ms) to be added to the beginning and end of each 
        simulation
    stim_type: string
        if 'linear_increase', the stimulus is mimicked as a gradual (step-wise)
        increase in external input rate.  Otherwise, single step function.
    time_to_max: float or int
        time to reach max input rate for 'linear_increase' stimulus (in ms)
    exc_connectivity: string
        type of E --> E and E --> I connectivity ('clustered' or 'random')
    inh_connectivity: string
        type of I --> E and I --> I connectivity ('clustered' or 'random')
    exc_syn_weight_dist: string
        type of distribution from which to draw E --> E and E --> I nonzero
        weights.  If 'beta', draw from beta distribution.  Otherwise, draw
        from continuous uniform distribution.
    net_state: string
        If 'synchronous', choose inh synaptic time constants that support
        network spike-rate oscillations in response to the stimulus.
        Otherwise, use same time constants for inh and exc synapses.
    ext_SD: bool
        if True, apply synaptic adaptation to external input synapses after
        stimulus onset
    int_SD: bool
        if True, apply synaptic adaptation to all 'intracortical' synapses 
        after stimulus onset        
    generate_LFPs: bool
        if True, simulate LFPs as sums of synaptic currents to groups of
        exc nodes
    save_results: bool
        if True, save results for each trial
    num_trials: float or int
        number of trials to simulation
    crit_freq: float or tuple of floats 
        critical frequency for filtering membrane potentials 
        (e.g., (20.0, 100.0))
    filt_kind: string
        kind of filter to use for membrane potentials (e.g., 'band' for 
        bandpass) 
    run_new_net_sim: Bool
        if True, run new network simulations (for num_trials trials), even if
        results for these settings already exist
    get_new_CC_results_all_pairs: Bool
        if True, calculate CCs for all pairs starting with traces.
    master_folder_path: string
        full path of directory containing data, code, figures, etc.

    Returns
    -------
    None
        
    '''
    
    epochs = ['ongoing', 'transient', 'steady-state']
    model_version = exc_connectivity + '_' + net_state
    
    print ''    
    print '  model version: %s'%model_version    
    print '  node_group: %s'%node_group

    #check for saved dictionary of population results first
    pop_CC_dict_path = master_folder_path + '\\saved_intermediate_results\\'
    pop_CC_dict_path += 'model_CC\\'
    if not os.path.isdir(pop_CC_dict_path):
        os.mkdir(pop_CC_dict_path)
    pop_CC_dict_path += 'avg_CCs_for_model_version_%s'%model_version
    pop_CC_dict_path += '_node_group_%s_'%node_group
    pop_CC_dict_path += str(windows) + '_ms_windows_'
    pop_CC_dict_path += str(gaps) + '_ms_gaps_'
    pop_CC_dict_path += str(padding) + '_ms_padding_'
    pop_CC_dict_path += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
    pop_CC_dict_path += '.p'
    
    ### master_dict format ###
    
    # master_dict = {'CC_dict':CC_dict, 'bootstrap_ranges':bootstrap_ranges}
    # CC_dict = {'ongoing':[CC_c1, CC_c2, ... ], ... , 
    #    'steady-state':[CC_c1, ... ]}  where CC_ci is across-trial avg.
    
    ### check for saved master_dict (trial-avg CC and boots for all cells) ###   
    if os.path.isfile(pop_CC_dict_path) and get_new_CC_results_all_pairs == False:
        master_dict = pickle.load(open(pop_CC_dict_path, 'rb'))
        
    ### get a new master_dict if one doesn't exist ###
    else:
        print 'getting new CC_dict for plots'

        CC_dict = {}
        bootstrap_ranges = {}
        
        for epoch in epochs:
            CC_dict[epoch] = []
            bootstrap_ranges[epoch] = []     
            
        ### populate the dictionaries ###
        CC_dict_by_pair_all_trials = get_CC_all_pairs(node_group, N_E, N_I,
            tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
            V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
            V_syn_I, C_E, C_I, g_AE, g_AI, g_GE, g_GI, g_extE, g_extI, 
            g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
            tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
            tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
            num_nodes_to_save, num_pairs, num_LFPs, windows, gaps, padding, 
            stim_type, time_to_max, exc_connectivity, inh_connectivity,
            exc_syn_weight_dist, net_state, ext_SD, int_SD, 
            generate_LFPs, save_results, num_trials, crit_freq, filt_kind, 
            run_new_net_sim, get_new_CC_results_all_pairs,
            master_folder_path)
            
            
        #CC_dict_by_cell_all_trials = {cell_name:{'ongoing':[CC_trial1, 
        #    ..., CC_trialn], ..., 'steady-state':[CC_trial1,
        #    ..., CC_trialn]]}}

        for pair_name in CC_dict_by_pair_all_trials:
            CC_dict_for_pair = CC_dict_by_pair_all_trials[pair_name]

            for epoch in epochs:
                CCs_for_epoch = CC_dict_for_pair[epoch]
                avg_CC = numpy.mean(CCs_for_epoch)
                bootstrap = get_bootstrap(CCs_for_epoch, num_reps = 1000,
                    conf_level = 0.95, num_comparisons = 1) 
                
                ## add trial-avg CC and bootstrap to CC_dict
                CC_dict[epoch].append(avg_CC)
                bootstrap_ranges[epoch].append(bootstrap)

        ### save these dicts for future use ###
        master_dict = {}
        master_dict['CC_dict'] = CC_dict
        master_dict['bootstrap_ranges'] = bootstrap_ranges

        pickle.dump(master_dict, open(pop_CC_dict_path, 'wb'))

    CC_dict = master_dict['CC_dict']
    bootstrap_ranges = master_dict['bootstrap_ranges']    
    
    plot_trajectory(ax, fontproperties, CC_dict, bootstrap_ranges, 
        crit_freq, model_version)
        

