"""
copyright (c) 2016 Mahmood Hoseini

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.

"""

import matplotlib.pyplot as plt
import numpy as np, matplotlib.pylab as plt, time
import os
from network_simulation.network_structure import \
get_clustered_forward_backward_network, get_random_network

from network_simulation.ext_input import get_ext_input, \
get_ext_input_linear_increase

from numpy import transpose as npt
import pylab
plt.ion(); start_time = time.time()
import cPickle as pickle

y = (1, 1, 0)
g = (0, 1, 0)
b = (0.2, 0.6, 1)

def get_avg_over_windows(trace, window_steps):
    new_trace = np.zeros(int(float(len(trace))/float(window_steps)))
    j = 0
    while window_steps*(j+1) <= len(trace) - 1:
        start_index = window_steps*j
        end_index = window_steps*(j+1)
        new_trace[j] += np.mean(trace[start_index:end_index])
        j += 1
    return new_trace


def run_network_simulation(N_E = 800, N_I = 200, tau_Em = 50, tau_Im = 25,
    p_exc = 0.03, p_inh = 0.20, tau_ref_E = 10, tau_ref_I = 5,
    V_leak_E_min = -70, V_leak_I_min = -70, V_th = -40, V_reset = -59,
    V_peak = 0, V_syn_E = 50, V_syn_I = -68, C_E = 0.4, C_I = 0.2,
    g_AE = 3.0e-3, g_AI = 6.0e-3, g_GE = 30.0e-3, g_GI = 30.0e-3, 
    g_extE = 6.0e-3, g_extI = 6.0e-3, g_leak_E = 10.0e-3, g_leak_I = 5.0e-3,
    tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, tau_AId = 1.0, tau_AEl = 1.5, 
    tau_AEr = 0.2, tau_AEd = 1.0, tauR_intra = 2500.0, tauD_intra = 15.0, 
    tauR_extra = 2500.0, tauD_extra = 15.0, rateI = 65, rateE = 65, 
    stim_factor = 4.5, num_nodes_to_save = 20, num_LFPs = 8, 
    windows = [1000.0, 1000.0, 1000.0], 
    gaps = [200.0, 0.0, 200.0], padding = 500.0, 
    stim_type = 'linear_increase', time_to_max = 200, 
    exc_connectivity = 'clustered', inh_connectivity = 'random',
    exc_syn_weight_dist = 'beta', net_state = 'synchronous',
    ext_SD = True, int_SD = True, 
    generate_LFPs = True, save_results = True,
    num_trials = 15, 
    master_folder_path = 'F:\\correlated_variability_'):

    '''
    Run num_trials repetitions of the LIF network simulation.
    
    Parameters
    ----------
    
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
    num_LFPs: int
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
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
    '''
        
    path = master_folder_path + '/code/network_simulation/'
    os.chdir(path)
    
    #set inhibitory time constants s.t. network is either synchronous
    #(i.e., oscillatory) or asynchronous after stimulus onset
    if net_state == 'synchronous':
        tau_Fr = 1.5
        tau_Fd = 6.0
    else:
        tau_Fr = 0.2
        tau_Fd = 1.0

    ### get neuron numbers to save ###
    #from a group of of 100 neighboring nodes, get 20 neighbors (from the
    #geometric center), and 20 randomly-selected from the full 100.
    #from the full excitatory network, get 20 randomly-selected nodes.
    center_of_group = int((N_E/float(num_LFPs))/2.)
    start_of_group = int(center_of_group - num_nodes_to_save/2.)
    end_of_group = int(start_of_group + num_nodes_to_save)
    r1_center_neighbors = pylab.arange(start_of_group, end_of_group)
    r1_random = np.random.permutation(100)[:num_nodes_to_save]
    r_all_random = np.random.permutation(pylab.arange(0, 800))[:num_nodes_to_save] 

    #let the resting membrane potential be variable across nodes (as is
    #observed in experiment)    
    V_leak_I = V_leak_I_min+10*np.random.random((N_I, 1));
    V_leak_E = V_leak_E_min+10*np.random.random((N_E, 1));


    dt = 0.05 #Time step in ms
        
    t_max = sum(windows) + sum(gaps) + 2*padding #Total simulation time in mSec
    t_stim = padding + windows[0] + gaps[0] #Stimulus onset time in mSec
    ntsteps = t_max/dt; 
    stim_onset_index = int(float(t_stim)/dt)
    
    ### Network configuration and external inputs ###
    
    #get I --> E and I --> I random connectivity
    WIIo2, WEEo2, WIEo2, WEIo2 = get_random_network(N_I, N_E, p_inh, p_inh, 
        p_inh, p_inh) 

    if exc_connectivity == 'clustered':    
    #get E --> E and E --> I clustered connectivity
        WIIo1, WEEo, WIEo, WEIo1 = get_clustered_forward_backward_network(N_I, 
            N_E, N_I*p_exc, N_E*p_exc, p_rw = 0.01,
            exc_syn_weight_dist = exc_syn_weight_dist)
    else:
        WEEo, WIEo = WEEo2, WIEo2
               
    WIIo, WEIo = WIIo2, WEIo2

    ### Equations of "motion" ###
        
    for trial in range(1, num_trials + 1) :
        print 'trial num:', str(trial)
        
        #synaptic weights for external inputs
        tWI = np.ones((N_I, 1)); tWE = np.ones((N_E, 1));

        if stim_type == 'linear_increase':        
            ext_input_I, ext_input_E = get_ext_input_linear_increase(N_I, 
                N_E, rateI, rateE, stim_factor, t_max, t_stim, time_to_max, dt)
        else:
            ext_input_I, ext_input_E = get_ext_input(N_I, N_E, rateI, rateE, 
                stim_factor, t_max, t_stim, time_to_max, dt)
        
        WII, WIE, WEI, WEE = WIIo, WIEo, WEIo, WEEo
        
        V_I = np.zeros((N_I , ntsteps)) + V_reset;
        V_E = np.zeros((N_E , ntsteps)) + V_reset;
        
        X_FF = np.zeros((N_I, 2)); S_FF = np.zeros((N_I, ntsteps))
        X_AI = np.zeros((N_I, 2)); S_AI = np.zeros((N_I, ntsteps))
        X_extI = np.zeros((N_I, 2)); S_extI = np.zeros((N_I, ntsteps))
        X_extE = np.zeros((N_E, 2)); S_extE = np.zeros((N_E, ntsteps))
        X_AE = np.zeros((N_E, 2)); S_AE = np.zeros((N_E, ntsteps))
        X_FE = np.zeros((N_E, 2)); S_FE = np.zeros((N_E, ntsteps))
        spike_train_I = np.zeros((N_I, ntsteps)); 
        spike_train_E = np.zeros((N_E, ntsteps));
        SCI = np.zeros((N_I, ntsteps))
        SCE = np.zeros((N_E, ntsteps))
    
        I_e_tot = np.zeros(ntsteps) #total excitatory current time series
        I_e_ext_tot = np.zeros(ntsteps) #total excitatory current (from 
                                        #external inputs) time series
        I_i_tot = np.zeros(ntsteps) #total inhibitory current time series
    
        mtW, mW = np.zeros((2, ntsteps-1)), np.zeros((4, ntsteps-1));
        for t in range(1, int(t_max/dt)) : 
            
            # Fast interneurons              
            X_AI[:, 1] = X_AI[:, 0]*(1 - dt/(tau_AIr)) + \
                tau_Im*dt/(tau_AIr)*np.dot(WIE, 
                spike_train_E[:, t-int(tau_AIl/dt)])
            S_AI[:, t] = S_AI[:, t-1]*(1 - dt/tau_AId) + dt/tau_AId*X_AI[:, 0]
            X_FF[:, 1] = X_FF[:, 0]*(1 - dt/tau_Fr) + \
                tau_Im*dt/tau_Fr*np.dot(WII, 
                spike_train_I[:, t-int(tau_Fl/dt)])
            S_FF[:, t] = S_FF[:, t-1]*(1 - dt/tau_Fd) + dt/tau_Fd*X_FF[:, 0]
            X_extI[:, 1] = X_extI[:, 0]*(1 - dt/tau_AIr) + \
                tau_Im*dt/tau_AIr*tWI[:, 0]*ext_input_I[:, t]
            S_extI[:, t] = S_extI[:, t-1]*(1 - dt/tau_AId) + \
                dt/tau_AId*X_extI[:, 0]
        
            index  = np.where(sum(npt(spike_train_I[:, t-tau_ref_I/dt:t])) == 0)[0]
            SCI[index, t] =  (- g_extI*S_extI[index, t-1]*(V_I[index, t-1] - V_syn_E)\
                              - g_GI*S_FF[index, t-1]*(V_I[index, t-1] - V_syn_I)\
                              - g_AI*S_AI[index, t-1]*(V_I[index, t-1] - V_syn_E))     
       
      
            V_I[index, t] = V_I[index, t-1] + dt/C_I*(SCI[index, t] - \
                g_leak_I*(V_I[index, t-1] - V_leak_I[index, 0]))
            V_I[np.where(V_I[:, t] >= V_th), t] = V_peak
            spike_train_I[np.where(V_I[:, t] >= V_th), t] = 1
    
            #get total synaptic currents and spikes
            I_e_tot[t] += sum(dt*(-1*g_AI*S_AI[index, t-1]*(V_I[index, t-1] - \
                V_syn_E)))
            I_e_ext_tot[t] += sum(dt*(-1*g_extI*S_extI[index, t-1]*(V_I[index, 
                t-1] - V_syn_E)))
            I_i_tot[t] += sum(dt*(-1*g_GI*S_FF[index, t-1]*(V_I[index, t-1] - 
                V_syn_I)))        

            
            # Excitatory neurons
            X_AE[:, 1] = X_AE[:, 0]*(1 - dt/tau_AEr) + \
                tau_Em*dt/tau_AEr*np.dot(WEE, 
                spike_train_E[:, t-int(tau_AEl/dt)])
            S_AE[:, t] = S_AE[:, t-1]*(1 - dt/tau_AEd) + dt/tau_AEd*X_AE[:, 0]

            #slower I --> E synapses
            X_FE[:, 1] = X_FE[:, 0]*(1 - dt/(1.5*tau_Fr)) + \
                tau_Em*dt/(1.5*tau_Fr)*np.dot(WEI, 
                spike_train_I[:, t-int(tau_Fl/dt)])
            S_FE[:, t] = S_FE[:, t-1]*(1 - dt/tau_Fd) + dt/tau_Fd*X_FE[:, 0]
            X_extE[:, 1] = X_extE[:, 0]*(1 - dt/tau_AEr) + \
                tau_Em*dt/tau_AEr*tWE[:, 0]*ext_input_E[:, t]
            S_extE[:, t] = S_extE[:, t-1]*(1 - dt/tau_AEd) + \
                dt/tau_AEd*X_extE[:, 0]
        
            index  = np.where(sum(npt(spike_train_E[:, t-tau_ref_E/dt:t])) == 0)[0]
            SCE[index, t] = (- g_extE*S_extE[index, t-1]*(V_E[index, t-1] - V_syn_E)\
                             - g_GE*S_FE[index, t-1]*(V_E[index, t-1] - V_syn_I)\
                             - g_AE*S_AE[index, t-1]*(V_E[index, t-1] - V_syn_E))
            V_E[index, t] = V_E[index, t-1] + dt/C_E*(SCE[index, t] - \
                g_leak_E*(V_E[index, t-1] - V_leak_E[index, 0]))
           
            V_E[np.where(V_E[:, t] >= V_th), t] = V_peak
            spike_train_E[np.where(V_E[:, t] >= V_th), t] = 1       
            
            if ext_SD == True and t > stim_onset_index:
                # thalamocortical synapse recovery
                tWI = tWI + dt/tauR_extra*(np.ones((N_I, 1))-tWI); 
                tWI[:, 0] = tWI[:, 0] - tWI[:, 0]*ext_input_I[:, t]*dt/tauD_extra; 
                mtW[0, t-1] = np.mean(tWI);
    
                # thalamocortical synapse recovery                
                tWE = tWE + dt/tauR_extra*(np.ones((N_E, 1))-tWE); 
                tWE[:, 0] = tWE[:, 0] - tWE[:, 0]*ext_input_E[:, t]*dt/tauD_extra; 
                mtW[1, t-1] = np.mean(tWE);
    
            if int_SD == True:
                NZ = np.nonzero(spike_train_I[:, t])
                WII = WII + dt/tauR_intra*(WIIo - WII); 
                WII[:, NZ] = WII[:, NZ] - WII[:, NZ]*dt/tauD_intra; 
    
                mW[0, t-1] = np.mean(WII);
    
                WEI = WEI + dt/tauR_intra*(WEIo - WEI); 
                WEI[:, NZ] = WEI[:, NZ] - WEI[:, NZ]*dt/tauD_intra; 
                mW[1, t-1] = np.mean(WEI);
    
                NZ = np.nonzero(spike_train_E[:, t])
                WIE = WIE + dt/tauR_intra*(WIEo - WIE); 
                WIE[:, NZ] = WIE[:, NZ] - WIE[:, NZ]*dt/tauD_intra; 
                mW[2, t-1] = np.mean(WIE);
    
                WEE = WEE + dt/tauR_intra*(WEEo - WEE); 
                WEE[:, NZ] = WEE[:, NZ] - WEE[:, NZ]*dt/tauD_intra; 
                mW[3, t-1] = np.mean(WEE);
            
            X_FF[:, 0] = X_FF[:, 1]; X_FF[:, 1] = 0;
            X_AI[:, 0] = X_AI[:, 1]; X_AI[:, 1] = 0;
            X_AE[:, 0] = X_AE[:, 1]; X_AE[:, 1] = 0;
            X_FE[:, 0] = X_FE[:, 1]; X_FE[:, 1] = 0;
            X_extI[:, 0] = X_extI[:, 1]; X_extI[:, 1] = 0;
            X_extE[:, 0] = X_extE[:, 1]; X_extE[:, 1] = 0;
    
        
        if generate_LFPs:
            cells_per_trode = int(N_E/(float(num_LFPs)))
            LFPs = np.zeros((num_LFPs, ntsteps))
            for ii in range(num_LFPs) :
                if ii == num_LFPs - 1: 
                    LFPs[ii, :] = np.sum(SCE[750:49, :], axis=0)
                else:
                    LFPs[ii, :] = np.sum(SCE[ii*cells_per_trode:(ii+1)*cells_per_trode-1, :], axis=0)
                
        ### Save results ###
        if save_results:        
            save_path = master_folder_path +  '\\model_results\\' 
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            save_path += exc_connectivity + '_' + net_state
            if int_SD == False or ext_SD == False:
                save_path += '_no_SD'
            save_path += '\\'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            SI_save_path = save_path + 'synaptic_inputs\\'
            if not os.path.isdir(SI_save_path):
                os.mkdir(SI_save_path)
                
            #synaptic inputs to cells of interest (to be injected into "test"
            #neurons in another function)
            np.save(SI_save_path + 'SFF_trial'+str(trial)+'.npy', 
                S_FF[:40, :]) #inh gating variable for inputs to 40 inhibitories
            np.save(SI_save_path + 'SAI_trial'+str(trial)+'.npy', 
                S_AI[:40, :]) #exc gating variable for inputs to 40 inhibitories
            np.save(SI_save_path + 'SextI_trial'+str(trial)+'.npy', 
                S_extI[:40, :]) #gating variable for external inputs to 40 inhibitories
    
            np.save(SI_save_path + 'SAE_1_center_trial'+str(trial)+'.npy', 
                S_AE[r1_center_neighbors, :]) 
            np.save(SI_save_path + 'SAE_1_random_trial'+str(trial)+'.npy', 
                S_AE[r1_random, :]) 
            np.save(SI_save_path + 'SAE_all_random_trial'+str(trial)+'.npy', 
                S_AE[r_all_random, :])     
    
            np.save(SI_save_path + 'SGE_1_center_trial'+str(trial)+'.npy', 
                S_FE[r1_center_neighbors, :]) 
            np.save(SI_save_path + 'SGE_1_random_trial'+str(trial)+'.npy', 
                S_FE[r1_random, :]) 
            np.save(SI_save_path + 'SGE_all_random_trial'+str(trial)+'.npy', 
                S_FE[r_all_random, :])            
            
            np.save(SI_save_path + 'SextE_1_center_trial'+str(trial)+'.npy', 
                S_extE[r1_center_neighbors, :]) 
            np.save(SI_save_path + 'SextE_1_random_trial'+str(trial)+'.npy', 
                S_extE[r1_random, :]) 
            np.save(SI_save_path + 'SextE_all_random_trial'+str(trial)+'.npy', 
                S_extE[r_all_random, :]) 
    
            if generate_LFPs:
                LFP_save_path = save_path + '\\LFPs\\' 
                if not os.path.isdir(LFP_save_path):
                    os.mkdir(LFP_save_path)
                LFP_save_path += 'LFPs_trial'+str(trial)+'.npy'
                np.save(LFP_save_path, LFPs)        
            
        if trial == num_trials:
            # save spike times for final trial (to be used in model figure)
            spike_times_E = {}
            spike_times_I = {}
            for i in range(N_I):
                cell_key = 'cell_%s'%str(i)
                spike_times_I[cell_key] = []
                for t in range(int(t_max/dt)) :
                    if spike_train_I[i, t] == 1 :
                        spike_times_I[cell_key].append(t)
            for i in range(N_E) :
                cell_key = 'cell_%s'%str(i)
                spike_times_E[cell_key] = []
                for t in range(int(t_max/dt)) :
                    if spike_train_E[i, t] == 1 :
                        spike_times_E[cell_key].append(t)
            spike_times = {'spike_times_E':spike_times_E,
                    'spike_times_I':spike_times_I}
            st_save_path = save_path + 'spike_times_trial%s.p'%str(trial)
            pickle.dump(spike_times, 
                open(st_save_path, 'wb'))

        print "elapsed time:", str(round((time.time() - start_time)/60.0, 2)) + " mins"
   
    print "It took", str(round((time.time() - start_time)/60.0, 2)) + " mins to run"
