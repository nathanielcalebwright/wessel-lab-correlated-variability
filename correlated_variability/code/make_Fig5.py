'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

import pylab
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import urllib2
import matplotlib as mpl


from sub_fig_plotting.plot_LIF_network_spike_trains_membrane_potentials_LFP import \
plot_LIF_network_spike_trains_membrane_potentials_LFP as plot_model_traces

from sub_fig_plotting.plot_CC_trajectories_model_neurons import \
plot_CC_trajectories_model_neurons as plot_CC

from sub_fig_plotting.plot_stim_trace import plot_stim_trace

from analysis.report_CC_results_model_network import \
report_CC_results_model_network as report_CC


def make_Fig5(exc_connectivity1 = 'random', net_state1 = 'asynchronous',
        exc_connectivity2 = 'random', net_state2 = 'synchronous',
        exc_connectivity3 = 'clustered', net_state3 = 'synchronous',
        node_group = '1_random', N_E = 800, N_I = 200, tau_Em = 50, 
        tau_Im = 25, p_exc = 0.03, p_inh = 0.20, tau_ref_E = 10, tau_ref_I = 5,
        V_leak_E_min = -70, V_leak_I_min = -70, V_th = -40, V_reset = -59,
        V_peak = 0, V_syn_E = 50, V_syn_I = -68, C_E = 0.4, C_I = 0.2,
        g_AE1 = 2.5e-3, g_AE2 = 2.5e-3, g_AE3 = 3.0e-3, g_AI = 6.0e-3, 
        g_GE = 30.0e-3, g_GI = 30.0e-3, 
        g_extE = 6.0e-3, g_extI1 = 4.0e-3, g_extI2 = 4.0e-3, g_extI3 = 6.0e-3,
        g_leak_E = 10.0e-3, g_leak_I = 5.0e-3,
        tau_Fl = 1.5, tau_AIl = 1.5, tau_AIr = 0.2, tau_AId = 1.0, tau_AEl = 1.5, 
        tau_AEr = 0.2, tau_AEd = 1.0, 
        tauR_intra = 2500.0, tauD_intra = 15.0, tauR_extra = 2500.0, 
        tauD_extra = 15.0, rateI = 65, rateE = 65, stim_factor = 4.5, 
        num_nodes_to_save = 20, num_pairs = 40, num_LFPs = 8, 
        windows = [1000.0, 1000.0, 1000.0], gaps = [200.0, 0.0, 200.0], 
        padding_for_traces = 500.0, stim_type = 'linear_increase', 
        time_to_max = 200, inh_connectivity = 'random',
        exc_syn_weight_dist = 'beta', ext_SD = True, int_SD = True, 
        generate_LFPs = True, save_results = True, num_trials = 15, 
        crit_freq_V = (20.0, 100.0), filt_kind_V = 'band', 
        crit_freq_LFP = 100.0, filt_kind_LFP = 'low',
        V_mult_factor = 10000.0, LFP_mult_factor = -2.5,
        run_new_net_sim = False,
        get_new_CC_results_all_pairs = False,
        master_folder_path = 'E:\\correlated_variability_',
        showfig = True, savefig = False):

    '''
    Generate Figure 5 from the manuscript.  Also, calculate and report CC
    results for clustered network with no synaptic depression.
        
    Parameters
    ----------
    exc_connectivity1: string
        type of E --> E and E --> I connectivity in model network used in A
        and B ('clustered' or 'random')
    net_state1: string
        Network 'state' for model used in A and B.
        If 'synchronous', choose inh synaptic time constants that support
        network spike-rate oscillations in response to the stimulus.
        Otherwise, use same time constants for inh and exc synapses.
    exc_connectivity2: string
        exc connectivity in model network used in C and D
    net_state2: string
        Network 'state' for model used in C and D.
    exc_connectivity3: string
        exc connectivity in model network used in E and F
    net_state3: string
        Network 'state' for model used in E and F.
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
    g_AE1: float
        synaptic conductance for AMPA channels on exc nodes in microS
        for model network used in A and B
    g_AE2: float
        synaptic conductance for AMPA channels on exc nodes in microS
        for model network used in C and D
    g_AE3: float
        synaptic conductance for AMPA channels on exc nodes in microS
        for model network used in E and F
    g_AI: float 
        synaptic conductance for AMPA channels on inh nodes in microS
    g_GE: float 
        synaptic conductance for GABA channels on exc nodes in microS
    g_GI1: float 
        synaptic conductance for GABA channels on inh nodes in microS
        for model network used in A and B
    g_GI2: float 
        synaptic conductance for GABA channels on inh nodes in microS
        for model network used in C and D
    g_GI3: float 
        synaptic conductance for GABA channels on inh nodes in microS
        for model network used in E and F
    tau_Fl: float or int
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
    inh_connectivity: string
        type of I --> E and I --> I connectivity ('clustered' or 'random')
    exc_syn_weight_dist: string
        type of distribution from which to draw E --> E and E --> I nonzero
        weights.  If 'beta', draw from beta distribution.  Otherwise, draw
        from continuous uniform distribution.
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
    get_new_CC_results_all_pairs: Bool
        if True, calculate CCs for all pairs starting with traces.
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
    showfig: Bool
        if True (and if savefig == True), plot spike rasters, include
        network schematics, and show the figure.  Otherwise, just calculate
        CCs for each of the three networks (which is relatively fast if 
        simulations already complete).
    savefig: Bool
        if True, save the figure to the specified path


    Returns
    -------
    None
        
    '''

    width_fig = 5.0
    height_fig = 9.0
    padding = 0.5 #space (in inches) b/w subfigs and edges of figure
    interpadding = 0.4 #reference unit for padding b/w subfigs
    
    fonts = FontProperties(); fonts.set_weight('bold'); fonts.set_size(6)
    fontm = FontProperties(); fontm.set_weight('bold'); fontm.set_size(12)
    fontl = FontProperties(); fontl.set_weight('bold'); fontl.set_size(12)
    mpl.rcParams['mathtext.default'] = 'regular'

        
    #a1:spike rasters, V1-V2 (20-100 Hz), LFP (for model_version1)
    #a2:stim trace for a1
    #a3:model schematic
    #b:V-V CC trajectories
    #c1:spike rasters, V1-V2 (20-100 Hz), LFP (for model_version2)
    #c2:stim trace for c1
    #c3:model schematic
    #d:V-V CC trajectories
    #e1:spike rasters, V1-V2 (20-100 Hz), LFP (for model_version3)
    #e2:stim trace for e1
    #e3:model schematic
    #f:V-V CC trajectories
    
    #widths and heights of all subfigures IN INCHES
    width_traj = 1.2
    width_traces = width_fig - 2*padding - 1.5*interpadding - width_traj
    height_stim = 0.1    
    height_traces = (1./3.)*(height_fig - 2*padding - 3*interpadding - 3*height_stim)
    height_traj = (3./5.)*(height_traces + height_stim - interpadding)
    height_schematic = (2./5.)*(height_traces + height_stim - interpadding)
    width_schematic = width_traj
        
    
    #x and y positions of all subfigures and labels IN INCHES
    x_a1 = padding
    y_a1 = height_fig - padding - height_traces
    x_a2 = x_a1
    y_a2 = y_a1 - height_stim
    x_a3 = width_fig - padding - width_schematic
    x_b = width_fig - padding - width_traj 
    y_b = y_a2
    y_a3 = y_b + height_traj + 0.35*interpadding
    x_c1 = x_a1
    y_c1 = y_a2 - 1.5*interpadding - height_traces
    x_c2 = x_c1
    y_c2 = y_c1 - height_stim
    x_c3 = x_a3
    x_d = x_b
    y_d = y_c2
    y_c3 = y_d + height_traj + 0.35*interpadding
    x_e1 = x_a1
    y_e1 = padding + height_stim
    x_e2 = x_e1
    y_e2 = padding
    x_e3 = x_a3
    y_e3 = padding + height_traj + 0.35*interpadding
    x_f = x_e3
    y_f = padding

    
    x_label_a = x_a1 - 0.3
    y_label_a = y_a1 + height_traces - 0.25*interpadding
    x_label_b = x_b - 0.4
    y_label_b = y_b + height_traj
    x_label_c = x_c1 - 0.3
    y_label_c = y_c1 + height_traces - 0.25*interpadding
    x_label_d = x_label_b
    y_label_d = y_d + height_traj
    x_label_e = x_e1 - 0.3
    y_label_e = y_e1 + height_traces - 0.25*interpadding
    x_label_f = x_label_b
    y_label_f = y_f + height_traj
    
    #x and y positions of all subfigures and labels IN FRACTIONS OF FIG SIZE
    x_a1 = (x_a1/width_fig)
    y_a1 = (y_a1/height_fig)
    x_a2 = (x_a2/width_fig)
    y_a2 = (y_a2/height_fig)
    x_a3 = (x_a3/width_fig)
    y_a3 = (y_a3/height_fig)
    x_b = (x_b/width_fig)
    y_b = (y_b/height_fig)

    x_c1 = (x_c1/width_fig)
    y_c1 = (y_c1/height_fig)
    x_c2 = (x_c2/width_fig)
    y_c2 = (y_c2/height_fig)
    x_c3 = (x_c3/width_fig)
    y_c3 = (y_c3/height_fig)
    x_d = (x_d/width_fig)
    y_d = (y_d/height_fig)
    
    x_e1 = (x_e1/width_fig)
    y_e1 = (y_e1/height_fig)
    x_e2 = (x_e2/width_fig)
    y_e2 = (y_e2/height_fig)
    x_e3 = (x_e3/width_fig)
    y_e3 = (y_e3/height_fig)
    x_f = (x_f/width_fig)
    y_f = (y_f/height_fig)
    
    x_label_a = (x_label_a/width_fig)
    y_label_a = (y_label_a/height_fig)
    x_label_b = (x_label_b/width_fig)
    y_label_b = (y_label_b/height_fig)
    x_label_c = (x_label_c/width_fig)
    y_label_c = (y_label_c/height_fig)
    x_label_d = (x_label_d/width_fig)
    y_label_d = (y_label_d/height_fig)
    x_label_e = (x_label_e/width_fig)
    y_label_e = (y_label_e/height_fig)    
    x_label_f = (x_label_f/width_fig)
    y_label_f = (y_label_f/height_fig)
    
    #widths and heights of all subfigures IN FRACTIONS OF FIG SIZE
    width_traces = (width_traces/width_fig)
    height_traces = (height_traces/height_fig) 
    width_traj = (width_traj/width_fig)
    height_traj = (height_traj/height_fig)
    width_schematic = (width_schematic/width_fig)
    height_schematic = (height_schematic/height_fig)
    height_stim = (height_stim/height_fig)
    
    V_mult_factor = 10000.0
    LFP_mult_factor = -2.5

    fig = plt.figure(figsize = (width_fig, height_fig), dpi = 300)  
    
    ###################### Panel A1: spikes, Vs, LFP
    width = width_traces
    height = height_traces
    x_pos = x_a1
    y_pos = y_a1
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_a, y_label_a, 'A', fontproperties=fontl)
    ax1 = fig.add_axes(rect)
    if showfig == True or savefig == True:
        plot_model_traces(ax1, fonts, node_group, N_E, N_I,
            tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
            V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
            V_syn_I, C_E, C_I, g_AE1, g_AI, g_GE, g_GI, g_extE, g_extI1, 
            g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
            tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
            tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
            num_nodes_to_save, num_LFPs, windows, gaps, padding_for_traces, 
            stim_type, time_to_max, exc_connectivity1, inh_connectivity,
            exc_syn_weight_dist, net_state1, ext_SD, int_SD, 
            generate_LFPs, save_results, num_trials, crit_freq_V, 
            filt_kind_V, crit_freq_LFP, filt_kind_LFP, V_mult_factor, 
            LFP_mult_factor, run_new_net_sim,
            master_folder_path)
    
    ###################### Panel A2: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_a2
    y_pos = y_a2
    rect = (x_pos, y_pos, width, height)
    ax2 = fig.add_axes(rect)
    plot_stim_trace(ax2, windows, gaps, 0)
    ax2.set_xlim(0, sum(windows) + sum(gaps))

    ###################### Panel A3: model_schematic
    width = width_schematic
    height = height_schematic
    x_pos = x_a3
    y_pos = y_a3
    rect = (x_pos, y_pos, width, height)
    ax3 = fig.add_axes(rect)   
    if showfig == True or savefig == True:
        image_path = 'file:\\' + master_folder_path + \
            '\\figures\\random_schematic.png'
        image = urllib2.urlopen(image_path)
        array = pylab.imread(image)
        ax3.imshow(array)    
        ax3.set_axis_off()
        ax3.text(0.5, 1.15, 'random asynchronous', fontsize = 8, 
            horizontalalignment = 'center',
            verticalalignment = 'bottom', transform = ax3.transAxes)
    
    ###################### Panel B: CC trajectories
    width = width_traj
    height = height_traj
    x_pos = x_b
    y_pos = y_b
    rect = (x_pos, y_pos, width, height)
    ax4 = fig.add_axes(rect)    
    fig.text(x_label_b, y_label_b, 'B', fontproperties=fontl)
    plot_CC(ax4, fonts, node_group, N_E, N_I,
        tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
        V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
        V_syn_I, C_E, C_I, g_AE1, g_AI, g_GE, g_GI, g_extE, g_extI1, 
        g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
        tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
        tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
        num_nodes_to_save, num_pairs, num_LFPs, windows, gaps, padding_for_traces, 
        stim_type, time_to_max, exc_connectivity1, inh_connectivity,
        exc_syn_weight_dist, net_state1, ext_SD, int_SD, 
        generate_LFPs, save_results, num_trials, crit_freq_V, filt_kind_V, 
        run_new_net_sim, get_new_CC_results_all_pairs,
        master_folder_path)

  
    ###################### Panel C1: spikes, Vs, LFP
    width = width_traces
    height = height_traces
    x_pos = x_c1
    y_pos = y_c1
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_c, y_label_c, 'C', fontproperties=fontl)
    ax5 = fig.add_axes(rect)
    if showfig == True or savefig == True:
        plot_model_traces(ax5, fonts, node_group, N_E, N_I,
            tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
            V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
            V_syn_I, C_E, C_I, g_AE2, g_AI, g_GE, g_GI, g_extE, g_extI2, 
            g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
            tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
            tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
            num_nodes_to_save, num_LFPs, windows, gaps, padding_for_traces, 
            stim_type, time_to_max, exc_connectivity2, inh_connectivity,
            exc_syn_weight_dist, net_state2, ext_SD, int_SD, 
            generate_LFPs, save_results, num_trials, crit_freq_V, 
            filt_kind_V, crit_freq_LFP, filt_kind_LFP, V_mult_factor, 
            LFP_mult_factor, run_new_net_sim,
            master_folder_path)

    
    ###################### Panel C2: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_c2
    y_pos = y_c2
    rect = (x_pos, y_pos, width, height)
    ax6 = fig.add_axes(rect)
    plot_stim_trace(ax6, windows, gaps, 0)
    ax6.set_xlim(0, sum(windows) + sum(gaps))

    ###################### Panel C3: model_schematic
    width = width_schematic
    height = height_schematic
    x_pos = x_c3
    y_pos = y_c3
    rect = (x_pos, y_pos, width, height)
    ax7 = fig.add_axes(rect)  
    if showfig == True or savefig == True:
        image_path = 'file:\\' + master_folder_path + \
            '\\figures\\random_schematic.png'
        image = urllib2.urlopen(image_path)
        array = pylab.imread(image)
        ax7.imshow(array)    
        ax7.set_axis_off()
        ax7.text(0.5, 1.15, 'random synchronous', fontsize = 8, 
            horizontalalignment = 'center',
            verticalalignment = 'bottom', transform = ax7.transAxes)
    
    ###################### Panel D: CC trajectories
    width = width_traj
    height = height_traj
    x_pos = x_d
    y_pos = y_d
    rect = (x_pos, y_pos, width, height)
    ax8 = fig.add_axes(rect)    
    fig.text(x_label_d, y_label_d, 'D', fontproperties=fontl)   
    plot_CC(ax8, fonts, node_group, N_E, N_I,
        tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
        V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
        V_syn_I, C_E, C_I, g_AE2, g_AI, g_GE, g_GI, g_extE, g_extI2, 
        g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
        tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
        tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
        num_nodes_to_save, num_pairs, num_LFPs, windows, gaps, padding_for_traces, 
        stim_type, time_to_max, exc_connectivity2, inh_connectivity,
        exc_syn_weight_dist, net_state2, ext_SD, int_SD, 
        generate_LFPs, save_results, num_trials, crit_freq_V, filt_kind_V, 
        run_new_net_sim, get_new_CC_results_all_pairs,
        master_folder_path)

    ###################### Panel E1: spikes, Vs, LFP
    width = width_traces
    height = height_traces
    x_pos = x_e1
    y_pos = y_e1
    rect = (x_pos, y_pos, width, height)
    fig.text(x_label_e, y_label_e, 'E', fontproperties=fontl)
    ax9 = fig.add_axes(rect)
    rateI = 50
    rateE = 50
    stim_factor = 6
    if showfig == True or savefig == True:
        plot_model_traces(ax9, fonts, node_group, N_E, N_I,
            tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
            V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
            V_syn_I, C_E, C_I, g_AE3, g_AI, g_GE, g_GI, g_extE, g_extI3, 
            g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
            tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
            tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
            num_nodes_to_save, num_LFPs, windows, gaps, padding_for_traces, 
            stim_type, time_to_max, exc_connectivity3, inh_connectivity,
            exc_syn_weight_dist, net_state3, ext_SD, int_SD, 
            generate_LFPs, save_results, num_trials, crit_freq_V, 
            filt_kind_V, crit_freq_LFP, filt_kind_LFP, V_mult_factor, 
            LFP_mult_factor, run_new_net_sim,
            master_folder_path)
                        
    
    ###################### Panel E2: stim trace
    width = width_traces
    height = height_stim
    x_pos = x_e2
    y_pos = y_e2
    rect = (x_pos, y_pos, width, height)
    ax10 = fig.add_axes(rect)
    plot_stim_trace(ax10, windows, gaps, 0)
    ax10.set_xlim(0, sum(windows) + sum(gaps))

    ###################### Panel E3: model_schematic
    width = width_schematic
    height = height_schematic
    x_pos = x_e3
    y_pos = y_e3
    rect = (x_pos, y_pos, width, height)
    ax11 = fig.add_axes(rect) 
    if showfig == True or savefig == True:
        image_path = 'file:\\' + master_folder_path + \
            '\\figures\\small_world_schematic.png'
        image = urllib2.urlopen(image_path)
        array = pylab.imread(image)
        ax11.imshow(array)    
        ax11.set_axis_off()
        ax11.text(0.5, 1.15, 'clustered', fontsize = 8, 
            horizontalalignment = 'center',
            verticalalignment = 'bottom', transform = ax11.transAxes)
    
    ###################### Panel F: CC trajectories
    width = width_traj
    height = height_traj
    x_pos = x_f
    y_pos = y_f
    rect = (x_pos, y_pos, width, height)
    ax12 = fig.add_axes(rect)    
    fig.text(x_label_f, y_label_f, 'F', fontproperties=fontl)  
    plot_CC(ax12, fonts, node_group, N_E, N_I,
        tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
        V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
        V_syn_I, C_E, C_I, g_AE3, g_AI, g_GE, g_GI, g_extE, g_extI3, 
        g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
        tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
        tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
        num_nodes_to_save, num_pairs, num_LFPs, windows, gaps, padding_for_traces, 
        stim_type, time_to_max, exc_connectivity3, inh_connectivity,
        exc_syn_weight_dist, net_state3, ext_SD, int_SD, 
        generate_LFPs, save_results, num_trials, crit_freq_V, filt_kind_V, 
        run_new_net_sim, get_new_CC_results_all_pairs,
        master_folder_path)

    ###################### Calculate and report CC values for clustered
    ###################### network with no synaptic depression
    ext_SD = False
    int_SD = False
    
    report_CC(node_group, N_E, N_I,
        tau_Em, tau_Im, p_exc, p_inh, tau_ref_E, tau_ref_I,
        V_leak_E_min, V_leak_I_min, V_th, V_reset, V_peak, V_syn_E, 
        V_syn_I, C_E, C_I, g_AE3, g_AI, g_GE, g_GI, g_extE, g_extI3, 
        g_leak_E, g_leak_I, tau_Fl, tau_AIl, tau_AIr, tau_AId, tau_AEl, 
        tau_AEr, tau_AEd, tauR_intra, tauD_intra, 
        tauR_extra, tauD_extra, rateI, rateE, stim_factor, 
        num_nodes_to_save, num_pairs, num_LFPs, windows, gaps, padding_for_traces, 
        stim_type, time_to_max, exc_connectivity3, inh_connectivity,
        exc_syn_weight_dist, net_state3, ext_SD, int_SD, 
        generate_LFPs, save_results, num_trials, crit_freq_V, filt_kind_V, 
        run_new_net_sim, get_new_CC_results_all_pairs,
        master_folder_path)    
    

    if savefig == True:
        figpath = master_folder_path + '\\figures\\Fig5'        
        fig.savefig(figpath + '.png', dpi = 300)    
        figpath2 = '\\home\\caleb\\Dropbox\\Fig5'
        fig.savefig(figpath2 + '.png', dpi = 300)    
    if showfig == True:
        pylab.show()

