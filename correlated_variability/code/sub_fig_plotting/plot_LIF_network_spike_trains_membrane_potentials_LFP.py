'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''
import matplotlib.pyplot as plt
import pylab
import numpy
import os
import cPickle as pickle
from signal_processing.filtering import butterworth as butter

from network_simulation.run_network_simulation import \
run_network_simulation as run_net_sim

from network_simulation.get_test_Vs_from_synaptic_inputs import \
get_test_Vs_from_synaptic_inputs as get_Vs

from network_simulation.get_downsampled_model_LFPs import \
get_downsampled_model_LFPs as get_LFPs


def plot_LIF_network_spike_trains_membrane_potentials_LFP(ax, fontproperties,
        node_group = '1_center', N_E = 800, N_I = 200, tau_Em = 50, 
        tau_Im = 25, p_exc = 0.03, p_inh = 0.20, tau_ref_E = 10, tau_ref_I = 5,
        V_leak_E_min = -70, V_leak_I_min = -70, V_th = -40, V_reset = -59,
        V_peak = 0, V_syn_E = 50, V_syn_I = -68, C_E = 0.4, C_I = 0.2,
        g_AE = 3.0e-3, g_AI = 6.0e-3, g_GE = 30.0e-3, g_GI = 30.0e-3, 
        g_extE = 6.0e-3, g_extI = 6.0e-3, g_leak_E = 10.0e-3, g_leak_I = 5.0e-3,
        tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, tau_AId = 1.0, tau_AEl = 1.5, 
        tau_AEr = 0.2, tau_AEd = 1.0, 
        tauR_intra = 2500.0, tauD_intra = 15.0, tauR_extra = 2500.0, 
        tauD_extra = 15.0, rateI = 65, rateE = 65, stim_factor = 4.5, 
        num_nodes_to_save = 20, num_LFPs = 8, 
        windows = [1000.0, 1000.0, 1000.0], 
        gaps = [200.0, 0.0, 200.0], padding = 500.0, 
        stim_type = 'linear_increase', time_to_max = 200, 
        exc_connectivity = 'clustered', inh_connectivity = 'random',
        exc_syn_weight_dist = 'beta', net_state = 'synchronous',
        ext_SD = True, int_SD = True, 
        generate_LFPs = True, save_results = True,
        num_trials = 15, crit_freq_V = (20.0, 100.0), filt_kind_V = 'band', 
        crit_freq_LFP = 100.0, filt_kind_LFP = 'low',
        V_mult_factor = 10000.0, LFP_mult_factor = -2.5,
        run_new_net_sim = False,
        master_folder_path = 'F:\\correlated_variability'):


    '''
    For a single trial from a given network model, plot spike rasters,
    a pair of high-frequency membrane potentials, and a single 'LFP'.  If
    necessary (or desired), run a new network simulation (num_trials times).
    
    Parameters
    ----------
    ax: matplotlib axis
    fontproperties: FontProperties object
        dictates size and font of text
    node_group: float
        group of network nodes from which to select example Vs and LFP
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
    tauR_intra: float
        synaptic weight recovery time constant in ms for intracortical
        synapses
    tauD_intra: float
        synaptic weight decay time constant in ms for intracortical synapses
    tauR_extra float
        synaptic weight recovery time constant in ms for external inputs
    tauD_extra: float
        synaptic weight decay time constant in ms for external inputs 
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
    crit_freq_V: float or tuple of floats 
        critical frequency for filtering membrane potentials 
        (e.g., (20.0, 100.0))
    filt_kind_V: string
        kind of filter to use for membrane potentials (e.g., 'band' for 
        bandpass)
    crit_freq_LFP: float or tuple of floats 
        critical frequency for filtering 'LFP' (e.g., 100.0) 
    filt_kind_LFP: string
        kind of filter to use for 'LFP' (e.g., 'low' for lowpass)
    V_mult_factor: int or float
        factor by which to multiply membrane potentials for example plot
        (e.g., 10000.0) 
    LFP_mult_factor: int or float
        factor by which to multiply 'LFP' for example plot (e.g., -2.5) 
    run_new_net_sim: Bool
        if True, run new network simulations (for num_trials trials), even if
        results for these settings already exist
    master_folder_path: string
        full path of directory containing data, code, figures, etc.

    Returns
    -------
    None
            
    '''

    fonts = fontproperties
    size = fonts.get_size()

    #background colors for each window of activity
    y = (1, 1, 0)
    g = (0, 1, 0)
    b = (0.2, 0.6, 1)

    start = int(padding)
    stop = int(padding + sum(windows) + sum(gaps))     

    data_path = master_folder_path + '\\model_results\\' + \
        exc_connectivity + '_' + net_state
    if int_SD == False:
        data_path += '_no_SD'
    data_path += '\\'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    
    ### plot spike trains ###

    spike_time_dict_path = data_path + 'spike_times_trial%s.p'%str(num_trials)
    if os.path.isfile(spike_time_dict_path) and run_new_net_sim == False:
        spike_time_dict = pickle.load(open(spike_time_dict_path, 'rb'))

    else:
        #run new network simulation if needed/desired
        run_net_sim(N_E, N_I, tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, 
            tau_ref_I, V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak,
            V_syn_E, V_syn_I, C_E, C_I, g_AE, g_AI, g_GE, g_GI, g_extE, g_extI, 
            g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
            tau_AEr, tau_AEd, tauR_intra, tauD_intra, tauR_extra, tauD_extra, 
            rateI, rateE, stim_factor, num_nodes_to_save, num_LFPs, windows, 
            gaps, padding, stim_type, time_to_max, exc_connectivity, 
            inh_connectivity, exc_syn_weight_dist, net_state, ext_SD, int_SD, 
            generate_LFPs, save_results, num_trials, master_folder_path)
            
        spike_time_dict = pickle.load(open(spike_time_dict_path, 'rb'))

    spike_times_E = spike_time_dict['spike_times_E']
    spike_times_I = spike_time_dict['spike_times_I']
    
    node_skips = 5 #number of nodes to skip when plotting spike spike trains
    
    #inh trains on bottom
    j = 0
    while j < N_I:
        cell_key = 'cell_%s'%str(j+1)
        AP_times = spike_times_I[cell_key] #in simulation time steps
        AP_times = 0.05*pylab.array(AP_times) #in ms
        for t in AP_times:
            ax.vlines(int(t - padding), j/node_skips, j/node_skips + 0.95, 
                color = 'b', lw = 0.25)
        j += node_skips

    #exc trains on bottom
    j = 0
    while j < N_E:
        cell_key = 'cell_%s'%str(j+1)
        AP_times = spike_times_E[cell_key]
        AP_times = 0.05*pylab.array(AP_times)
        for t in AP_times:
            ax.vlines(int(t - padding), j/node_skips + 200/node_skips, 
                j/node_skips + 200/node_skips + 0.95, color = 'k', lw = 0.25)    
        j += node_skips
    
    ### plot a pair of high-freq membrane potentials ###

    V_dict = get_Vs(node_group, N_E, N_I, tau_Em, tau_Im, p_exc, p_inh, 
        tau_ref_E, tau_ref_I, V_leak_E_min, V_leak_I_min, V_th, 
        V_reset, V_peak, V_syn_E, V_syn_I, C_E, C_I, g_AE, g_AI, g_GE, 
        g_GI, g_extE, g_extI, g_leak_E, g_leak_I, tau_Fl, tau_AIl, 
        tau_AIr, tau_AId, tau_AEl, tau_AEr, tau_AEd, tauR_intra, 
        tauD_intra, tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
        num_nodes_to_save, num_LFPs, windows, gaps, padding, stim_type, 
        time_to_max, exc_connectivity, inh_connectivity, 
        exc_syn_weight_dist, net_state, ext_SD, int_SD, 
        generate_LFPs, save_results, num_trials, crit_freq_V, filt_kind_V,
        run_new_net_sim, master_folder_path)

    
    V1 = V_mult_factor*V_dict['cell_01'][int(num_trials - 1)]
    V2 = V_mult_factor*V_dict['cell_20'][int(num_trials - 1)]
    
    #filter
    V1 = butter(V1, sampling_freq = 1000.0, critical_freq = crit_freq_V,
            order=3, kind = filt_kind_V)    
    V2 = butter(V2, sampling_freq = 1000.0, critical_freq = crit_freq_V,
            order=3, kind = filt_kind_V) 

    #trim off the ends and plot   
    ax.plot(V1[start:stop] - 60, 'k', lw = 0.5)
    ax.plot(V2[start:stop] - 60, 'r', lw = 0.5) 

    ### plot LFP ###

    LFP_dict = get_LFPs(exc_connectivity, net_state, int_SD, num_trials,
        master_folder_path)   
    
    
    #get LFP from geometric center of network (if clustered)
    LFP = LFP_mult_factor*LFP_dict['LFP_4']
    
    #filter
    LFP = butter(LFP, sampling_freq = 1000.0, critical_freq = crit_freq_LFP,
            order=3, kind = filt_kind_LFP)     
    
    #trim off the ends and plot
    ax.plot(LFP[start:stop] - 100, 'k', lw = 0.5)

    #indicate epoch windows with color        
    ong_start = 0
    ong_stop = windows[0]
    trans_start = ong_stop + gaps[0] + gaps[1]
    trans_stop = trans_start + windows[1]
    ss_start = trans_stop + gaps[2]
    ss_stop = ss_start + windows[2]
    
    ax.axvspan(ong_start, ong_stop, ymin = 0, ymax = 1, 
        facecolor = y, ec = 'none', alpha = 0.3)
    
    ax.axvspan(trans_start, trans_stop, 
        ymin = 0, ymax = 1, facecolor = b, ec = 'none', alpha = 0.3)
    
    ax.axvspan(ss_start, ss_stop, 
        ymin = 0, ymax = 1, facecolor = g, ec = 'none', alpha = 0.3)                 
                            
    #configure the axis
    ax.set_xlim(0, (sum(windows) + sum(gaps)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #add scale bars
    V_bar_xs = [500, 500]
    V_bar_ys = [V1[0] - 60 + 30, V1[0] - 60 + 50]
    V_text1 = '2 mV'
    V_text2 = '2 a.u.'
    
    t_bar_xs = [V_bar_xs[0], V_bar_xs[0] + 500]
    t_bar_ys = [V_bar_ys[0], V_bar_ys[0]]
    V_text_y1 = V_bar_ys[1]
    V_text_y2 = V_bar_ys[0]
    t_text_y = t_bar_ys[0] + 5

    ax.plot(V_bar_xs, V_bar_ys, 'k', lw = 1.5)
    ax.plot(t_bar_xs, t_bar_ys, 'k', lw = 1.5)
    
    if exc_connectivity == 'random' and net_state == 'asynchronous':    
        ax.text(V_bar_xs[0] - 50, V_text_y1, V_text1, fontsize = size,
            horizontalalignment = 'right', verticalalignment = 'center')
    
        ax.text(V_bar_xs[0] - 50, V_text_y2, V_text2, fontsize = size,
            horizontalalignment = 'right', verticalalignment = 'bottom')
    
        ax.text(numpy.mean(t_bar_xs) + 25, t_text_y, '500 ms', fontsize = size,
            horizontalalignment = 'center', verticalalignment = 'bottom')
    
    #label V1, V2, and "LFP"
    if exc_connectivity == 'random' and net_state == 'asynchronous':    
        V1_text_x = 2600
        V2_text_x = 2850
        LFP_text_x = V1_text_x
        
        V1_text_y = t_text_y
        V2_text_y = V1_text_y
        LFP_text_y = LFP[LFP_text_x] - 100 + 30
        
        ax.text(V1_text_x, V1_text_y, 'V1, ', color = 'k', fontsize = size)
        ax.text(V2_text_x, V2_text_y, 'V2', color = 'r', fontsize = size)
        ax.text(LFP_text_x, LFP_text_y, '"LFP"', color = 'k', fontsize = size)
    

    

    
    
