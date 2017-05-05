"""
copyright (c) 2015 Mahmood Hoseini

This program is free software: you can redistribute it and/or modify it
under the terms of the gnu general public license as published by the
free software foundation, either version 3 of the license, or (at your
option) any later version. you should have received a copy of the gnu
general public license along with this program. If not, see
<http://www.gnu.org/licenses/>.

Leaky-integrate-and-fire neuron.

"""
import numpy as np
import pylab
import os
import cPickle as pickle
from signal_processing.downsample_trace import downsample_trace as downsamp
import matplotlib.pyplot as plt
from network_simulation.run_network_simulation import run_network_simulation \
as run_net_sim

def generate_test_V(path, cluster_name, cell_index, trial, tau_Em, V_rest, 
        V_syn_E = 50, V_syn_I = -68, g_AE = 3e-3, g_AI = 6e-3, 
        g_GE = 30e-3, g_GI = 30e-3, g_extE = 6e-3, g_extI = 6e-3, 
        g_leak = 10e-3, C_E = 0.4):
    
    dt = 0.00005 #time step of simulation in s   
    mult_factor = 0.5 #scaling factor for synaptic inputs to "test" neuron  
    
    #get everything in units of s, S, F, V  
    tau_Em *= 1e-3
    g_AE *= 1e-6
    g_AI *= 1e-6
    g_GE *= 1e-6
    g_GI *= 1e-6
    g_extE *= 1e-6
    g_extI *= 1e-6
    g_leak *= 1e-6
    V_syn_E *= 1e-3
    V_syn_I *= 1e-3
    C_E *= 1e-9
    Rm = tau_Em/C_E #make Rm/tau_Em = 1/C_E from network sim

    ## Loading data
    Ne = 800
    if cell_index < Ne :
        SAE = np.load(path + 'SAE'+ '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        SGE = np.load(path + 'SGE'+ '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        SextE = np.load(path + 'SextE' + '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        
        g_exc = mult_factor*g_AE*SAE[cell_index, :];
        g_inh = mult_factor*g_GE*SGE[cell_index, :];
        g_ext = mult_factor*g_extE*SextE[cell_index, :];
    else :
        SFF = np.load(path + 'SFF'+ '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        SAI = np.load(path + 'SAI'+ '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        SextI = np.load(path + 'SextI'+ '_' + cluster_name + \
            '_trial%s'%str(trial)+'.npy')
        
        g_exc = mult_factor*g_AI*SAI[cell_index-Ne, :];
        g_inh = mult_factor*g_GI*SFF[cell_index-Ne, :];
        g_ext = mult_factor*g_extI*SextI[cell_index-Ne, :];


    nsteps = len(g_exc);
    V = V_rest*np.ones(np.size(g_exc));

    for t in range(2, nsteps) :
        V[t] = V[t-1] - dt/tau_Em*(Rm*g_leak*(V[t-1] - V_rest) + \
            Rm*(g_exc[t-1]+g_ext[t-1])*(V[t-1] - V_syn_E) + \
            Rm*g_inh[t-1]*(V[t-1] - V_syn_I))
                
    #downsample V to 1 kHz
    downsamp_freq = 1000.0
    samp_freq = 20000.0
    ndx_skips = int(samp_freq/downsamp_freq)
    V = downsamp(V, ndx_skips = ndx_skips)
    
    return V    
    
def get_test_Vs_from_synaptic_inputs(node_group = '1_center', N_E = 800, 
        N_I = 200, tau_Em = 50, tau_Im = 25, p_exc = 0.03, p_inh = 0.20, 
        tau_ref_E = 10, tau_ref_I = 5, V_leak_E_min = -70, V_leak_I_min = -70, 
        V_th = -40, V_reset = -59, V_peak = 0, V_syn_E = 50, V_syn_I = -68, 
        C_E = 0.4, C_I = 0.2, g_AE = 3.0e-3, g_AI = 6.0e-3, g_GE = 30.0e-3, 
        g_GI = 30.0e-3, g_extE = 6.0e-3, g_extI = 6.0e-3, g_leak_E = 10.0e-3,
        g_leak_I = 5.0e-3, tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, 
        tau_AId = 1.0, tau_AEl = 1.5, 
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
        num_trials = 15, crit_freq = (20.0, 100.0), filt_kind = 'band', 
        run_new_net_sim = False,
        master_folder_path = 'F:\\correlated_variability_'):


    '''
    Generate 'test' membrane potentials by injecting synaptic currents 
    generated by network simulation into single-compartment LIF neurons.
    Synaptic inputs are multiplied by mult_factor (ideally < 1), and the
    spike mechanism has been removed, to avoid spiking in test neurons.
    
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
    crit_freq: float or tuple of floats 
        critical frequency for filtering membrane potentials 
        (e.g., (20.0, 100.0))
    filt_kind: string
        kind of filter to use for membrane potentials (e.g., 'band' for 
        bandpass) 
    run_new_net_sim: Bool
        if True, run new network simulations (for num_trials trials), even if
        results for these settings already exist
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    
    V_dict: python dictionary of membrane potential traces (downsampled to 
        1 kHz) for all test neurons of interest.
    
    '''    

    save_path = master_folder_path +  '\\model_results\\' 
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path += exc_connectivity 
    save_path += '_' + net_state
    if int_SD == False:
        save_path += '_no_SD'
    save_path += '\\'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    V_save_path = save_path + 'downsamp_membrane_potentials\\'
    if not os.path.isdir(V_save_path):
        os.mkdir(V_save_path)
    V_save_path += node_group + '.p'  
    if os.path.isfile(V_save_path):
        V_dict = pickle.load(open(V_save_path, 'rb'))
    else:    
        SI_save_path = save_path + 'synaptic_inputs/'
        if not os.path.isdir(SI_save_path):
            os.mkdir(SI_save_path)
            
        node_groups = ['all_random', '1_random', '1_center']
        
        #make sure saved synaptic inputs exist
        check_path = SI_save_path + 'SAE_' + node_groups[0] \
            + '_trial%s' %str(num_trials)+ '.npy'

        if os.path.isfile(check_path) and run_new_net_sim == False:
            run_sim = False
        else:
            run_sim = True
        
        if run_sim:
            run_net_sim(N_E, N_I, tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, 
                tau_ref_I, V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak,
                V_syn_E, V_syn_I, C_E, C_I, g_AE, g_AI, g_GE, g_GI, g_extE, g_extI, 
                g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
                tau_AEr, tau_AEd, tauR_intra, tauD_intra, tauR_extra, tauD_extra, 
                rateI, rateE, stim_factor, num_nodes_to_save, num_LFPs, windows, 
                gaps, padding, stim_type, time_to_max, exc_connectivity, 
                inh_connectivity, exc_syn_weight_dist, net_state, ext_SD, int_SD, 
                generate_LFPs, save_results, num_trials, master_folder_path)
        
        for node_group in node_groups:
            print 'generating membrane potentials for %s'%node_group
            V_dict = {}
    
            for j in range(20):
                V_all_trials = []
                cell_num = j + 1
                if cell_num < 10:
                    cell_name = 'cell_0%s'%str(cell_num)
                else:
                    cell_name = 'cell_%s'%str(cell_num)
                Vrest = (-70+10*np.random.random())*1e-3
                print '  ', cell_name
                for trial in range(num_trials) :
                    print '    trial num', trial + 1
                    V_for_trial = generate_test_V(SI_save_path, 
                        node_group, j, trial+1, tau_Em, Vrest, V_syn_E, V_syn_I, 
                        g_AE, g_AI, g_GE, g_GI, g_extE, g_extI, g_leak_E, 
                        C_E)                  
                    
                    V_all_trials.append(V_for_trial)
                        
                V_dict[cell_name] = pylab.array(V_all_trials)
    
            V_path = save_path + 'downsamp_membrane_potentials\\' \
                + node_group + '.p'
            pickle.dump(V_dict, open(V_path, 'wb'))  
      
    return V_dict
    
    
