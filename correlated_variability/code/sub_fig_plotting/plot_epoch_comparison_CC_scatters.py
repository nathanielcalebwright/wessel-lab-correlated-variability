import matplotlib.ticker as plticker
import numpy
from scipy.stats import linregress
import cPickle as pickle
import os
import matplotlib as mpl


from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

from statistical_analysis.singer_correlation_comparisons import dependent_corr

from analysis.get_CC_all_pairs import get_CC_all_pairs

def plot_scatter(ax, fontproperties, master_dict, epoch_pair):
    
    fonts = fontproperties
    size = fonts.get_size()
    mpl.rcParams['mathtext.default'] = 'regular'


    CCs_by_epoch = master_dict['CC_dict']
    boots_by_epoch = master_dict['bootstrap_ranges']
    
    CCs_x = CCs_by_epoch[epoch_pair[0]]
    CCs_y = CCs_by_epoch[epoch_pair[1]]
                         
    ### make scatter for across-trial averages ###        
    x_min = round(min(CCs_x) - 0.06, 1) - 0.001
    x_max = round(max(CCs_x) + 0.06, 1) + 0.001
    y_min = round(min(CCs_y) - 0.06, 1) - 0.001
    y_max = round(max(CCs_y) + 0.06, 1) + 0.001
   
    
    ax.plot(CCs_x, CCs_y, 'o',
        color = 'k', markersize = 4, mec = 'k', mew = 0.45, 
        alpha = 0.45, clip_on = False)
                        
    slope, intercept, r, p, std_err = linregress(CCs_x, 
        CCs_y)
    p_corrected = 3*p #correct for three comparisons (2 in this figure, one in Fig 4)
    

    print '    %s CCs'%epoch_pair[0],  'vs. %s CCs'%epoch_pair[1]
    print '      lin_regress r = %s'%str(r), '(P_corrected = %s)'%str(p_corrected)
    
    #print fit results on axis
    r_string = str(round(r**2, 2))
    p_string = str(round(p, 3))
    
    fit_string = 'r = %s'%r_string + '\np = %s'%p_string
    ax.text(0.5, 0.875, fit_string, fontsize = size, 
        horizontalalignment = 'center', 
        verticalalignment = 'bottom',
        transform = ax.transAxes)
        
    fit_ys = []
    fit_xs = []
    x = min(CCs_x)
    while x <= max(CCs_x):
        fit_xs.append(x)
        y_ = slope*x + intercept
        fit_ys.append(y_)
        x += 0.01
    
    ax.plot(fit_xs, fit_ys, color = 'r', lw = 1.5)                       
    ax.set_xlabel('%s CC'%epoch_pair[0], fontsize = size)
    y_label = '%s CC'%epoch_pair[1]
    ax.set_ylabel(y_label, fontsize = size)
    text = 'r = %s'%str(round(r, 3)) + ' (p = %s)'%str(round(p, 4))
    ax.tick_params('both', length = 3, width = 1.0, which = 'major',
        direction = 'outward')    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    locy = plticker.MultipleLocator(base = 0.1)

    locx = plticker.MultipleLocator(base = 0.1)
    ax.yaxis.set_major_locator(locy)
    ax.xaxis.set_major_locator(locx)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    y_labels = ax.get_yticks()
    x_labels = ax.get_xticks()
    ax.set_yticklabels(y_labels, fontsize = size)
    ax.set_xticklabels(x_labels, fontsize = size) 
    
    return r

        

def plot_epoch_comparison_CC_scatters(ax1, ax2, ax3,
    fontproperties, stim_types = ['extended_stim', 'flash'],
    windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'band',
    replace_spikes = True, remove_sine_waves = True,
    get_new_CC_results_all_pairs = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):
    
    '''
    Compare the CC values for all possible pairs of epochs.
    
    Calculate the trial-averaged Pearson correlation coefficient (CC) for each
    cell in the dataset, for the ongoing, transient, and steady-state epochs, 
    for the crit_freq freq band.  Plot the results in "trajectory" view.  For 
    each pair, the significance of the across-epoch change is assessed by
    bootstrapping.  For the population, the significance of the across-epoch
    change is assessed via the Wilcoxon signed-rank test.
    
    Parameters
    ----------
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

    epochs = ['ongoing', 'transient', 'steady-state']
        

    #check for saved dictionary of population results first
    pop_CC_dict_path = master_folder_path + '\\saved_intermediate_results\\'
    pop_CC_dict_path += 'CC\\avg_CCs_for_population_'

    for stim_type in stim_types:                
        pop_CC_dict_path += '%s_'%stim_type
    pop_CC_dict_path += str(windows) + '_ms_windows_'
    pop_CC_dict_path += str(gaps) + '_ms_gaps_'
    pop_CC_dict_path += str(padding) + '_ms_padding_'
    pop_CC_dict_path += str(crit_freq) + '_Hz_%spass_filt'%filt_kind
    if replace_spikes:
        pop_CC_dict_path += '_spikes_removed'
    if remove_sine_waves:
        pop_CC_dict_path += '_sine_removed'
    pop_CC_dict_path += '.p'
    
    ### master_dict format ###
    
    # master_dict = {'CC_dict':CC_dict, 'bootstrap_ranges':bootstrap_ranges}
    # CC_dict = {'ongoing':[CC_pr1, CC_pr2, ... ], ... , 
    #    'steady-state':[CC_pr1, ... ]}  where CC_pri is across-trial avg.
    
    ### check for saved master_dict (trial-avg CC and boots for all cells) ###   
    if os.path.isfile(pop_CC_dict_path) and get_new_CC_results_all_pairs == False:
        master_dict = pickle.load(open(pop_CC_dict_path, 'rb'))
        
    ### get a new master_dict if one doesn't exist ###
    else:
        CC_dict = {}
        bootstrap_ranges = {}
        
        for epoch in epochs:
            CC_dict[epoch] = []
            bootstrap_ranges[epoch] = []               

        ### populate the dictionaries ###
        CC_dict_by_cell_all_trials = get_CC_all_pairs(stim_types,
            windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
            remove_sine_waves, get_new_CC_results_all_pairs,
            get_new_processed_traces, master_folder_path) 

        
        #CC_dict_by_cell_all_trials = {cell_name:{'ongoing':[CC_trial1, 
        #    ..., CC_trialn], ..., 'steady-state':[CC_trial1,
        #    ..., CC_trialn]]}}

        for cell_name in CC_dict_by_cell_all_trials:
            CC_dict_for_cell = CC_dict_by_cell_all_trials[cell_name]

            for epoch in epochs:
                CCs_for_epoch = CC_dict_for_cell[epoch]
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
            
    r1 = plot_scatter(ax1, fontproperties, master_dict, 
            ['ongoing', 'transient'])
    r2 = plot_scatter(ax2, fontproperties, master_dict, 
            ['transient', 'steady-state'])
    r3 = plot_scatter(ax3, fontproperties, master_dict, 
            ['ongoing', 'steady-state'])

    #get p-val for comparison of three r-values
    t_comp, p_comp = dependent_corr(r1, r2, r3, 
        len(master_dict['CC_dict']['ongoing']))

    print '\n    # comparison of r-values #'
    print '      t = %s'%str(t_comp)
    print '      p = %s'%str(p_comp)

                

    


