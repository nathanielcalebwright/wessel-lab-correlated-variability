'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

from general.get_cell_pairs_for_model_analysis import \
get_cell_pairs_for_model_analysis

from signal_processing.filtering import butterworth as butter

from network_simulation.get_test_Vs_from_synaptic_inputs import \
get_test_Vs_from_synaptic_inputs as get_Vs


import os
import cPickle as pickle
import numpy
from scipy.stats import pearsonr
import pylab

def get_CC_all_pairs_model_network(node_group = '1_center', N_E = 800, 
        N_I = 200, tau_Em = 50, tau_Im = 25, p_exc = 0.03, p_inh = 0.20, 
        tau_ref_E = 10, tau_ref_I = 5, V_leak_E_min = -70, V_leak_I_min = -70,
        V_th = -40, V_reset = -59, V_peak = 0, V_syn_E = 50, V_syn_I = -68, 
        C_E = 0.4, C_I = 0.2, g_AE = 3.0e-3, g_AI = 6.0e-3, g_GE = 30.0e-3, 
        g_GI = 30.0e-3, g_extE = 6.0e-3, g_extI = 6.0e-3, g_leak_E = 10.0e-3, 
        g_leak_I = 5.0e-3, tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, 
        tau_AId = 1.0, tau_AEl = 1.5, tau_AEr = 0.2, tau_AEd = 1.0, 
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
    
    
    '''
    For each pair test neuron pair, get the Pearson correlation coefficient of 
    the membrane potentials for each trial and epoch.
    
    Parameters
    ----------
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
    CC_dict_all_pairs: python dictionary of CC values for all trials and 
    epochs, for each pair of interest.
    
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
    model_version = exc_connectivity + '_' + net_state
     
    CC_dict_path = master_folder_path + '\\saved_intermediate_results\\'
    if not os.path.isdir(CC_dict_path):
        os.mkdir(CC_dict_path)
    CC_dict_path += 'model_CC\\'
    if not os.path.isdir(CC_dict_path):
        os.mkdir(CC_dict_path)
    CC_dict_path += 'CC_all_pairs_and_trials_model_version_%s'%model_version
    CC_dict_path += '_node_group_%s_'%node_group
    CC_dict_path += str(windows) + '_ms_windows_'
    CC_dict_path += str(gaps) + '_ms_gaps_'
    CC_dict_path += str(padding) + '_ms_padding_'
    CC_dict_path += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
    CC_dict_path += '.p'
    
    ### use saved pwr dictionary, if it exists ###
    if os.path.isfile(CC_dict_path) and get_new_CC_results_all_pairs == False:
        CC_dict_all_pairs = pickle.load(open(CC_dict_path, 'rb'))

    ### or make a new CC dictionary ###
    else:
        CC_dict_all_pairs = {}
        
        model_pair_path = master_folder_path + \
            '\\code\\general\\model_pairs.npy'
        if not os.path.isfile(model_pair_path):
            pairs = get_cell_pairs_for_model_analysis(num_nodes_to_save, 
                num_pairs)
            numpy.save(model_pair_path, pairs)
        else:
            pairs = numpy.load(model_pair_path)
            #pairs are randomly-generated, but want to use same pairs
            #for all model analysis

        ### get raw traces, filter, discard padding on ends ###
        V_dict_path = master_folder_path + \
            '\\model_results\\%s\\membrane_potentials\\'%model_version
        V_dict_path += '%s'%node_group
        V_dict_path += '.p'
        if os.path.isfile(V_dict_path) and run_new_net_sim == False:
            V_dict = pickle.load(open(V_dict_path, 'rb'))
        else:
            V_dict = get_Vs(node_group, N_E, N_I, tau_Em, tau_Im, p_exc, p_inh, 
                tau_ref_E, tau_ref_I, V_leak_E_min, V_leak_I_min, V_th, 
                V_reset, V_peak, V_syn_E, V_syn_I, C_E, C_I, g_AE, g_AI, g_GE, 
                g_GI, g_extE, g_extI, g_leak_E, g_leak_I, tau_Fl, tau_AIl, 
                tau_AIr, tau_AId, tau_AEl, tau_AEr, tau_AEd, tauR_intra, 
                tauD_intra, tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
                num_nodes_to_save, num_LFPs, windows, gaps, padding, stim_type, 
                time_to_max, exc_connectivity, inh_connectivity, 
                exc_syn_weight_dist, net_state, ext_SD, int_SD, 
                generate_LFPs, save_results, num_trials, crit_freq, filt_kind,
                run_new_net_sim, master_folder_path)
                
        for pair in pairs:
            cell_name1 = pair[0]
            cell_name2 = pair[1]   
            pair_name = cell_name1 + '-' + cell_name2
            CC_dict_all_pairs[pair_name] = {}
            for epoch in epochs:
                CC_dict_all_pairs[pair_name][epoch] = []
                
            #process the traces (filter, discard padding)
            traces1 = V_dict[cell_name1]
            traces2 = V_dict[cell_name2]

            proc_traces1 = []
            proc_traces2 = []         
            for k, subtrace in enumerate(traces1):                
                #filter traces
                trace1 = butter(traces1[k], critical_freq = crit_freq,
                    kind = filt_kind, order = 3, sampling_freq = 1000.0)
                trace2 = butter(traces2[k], critical_freq = crit_freq,
                    kind = filt_kind, order = 3, sampling_freq = 1000.0)

                #discard padding from each end
                trace1 = trace1[sub_ong_start:sub_ss_stop]
                trace2 = trace2[sub_ong_start:sub_ss_stop]
                proc_traces1.append(trace1)
                proc_traces2.append(trace2)


            proc_traces1 = pylab.array(proc_traces1)
            proc_traces2 = pylab.array(proc_traces2)

            for k, trace in enumerate(proc_traces1):
                #get residual traces for each trial
                resid1 = proc_traces1[k] - numpy.mean(proc_traces1, axis = 0)
                resid2 = proc_traces2[k] - numpy.mean(proc_traces2, axis = 0)
                
                #calculate CC (or Pearson r) for each epoch
                for j, epoch in enumerate(epochs):
                    start = sub_intervals[j][0]
                    end = sub_intervals[j][1]
                    epoch_resid1 = resid1[start:end]
                    epoch_resid2 = resid2[start:end]
                    CC = pearsonr(epoch_resid1, epoch_resid2)[0]
                    CC_dict_all_pairs[pair_name][epoch].append(CC)
        
        pickle.dump(CC_dict_all_pairs, open(CC_dict_path, 'wb'))

    return CC_dict_all_pairs

