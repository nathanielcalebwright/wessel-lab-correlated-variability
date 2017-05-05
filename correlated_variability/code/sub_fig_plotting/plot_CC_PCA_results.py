# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:29:02 2017

@author: Caleb
"""
from __future__ import division

import cPickle as pickle
import os
import scipy

import numpy as np
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import random
import pylab
import matplotlib.ticker as plticker
import numpy

from analysis.get_CC_all_pairs import get_CC_all_pairs

from statistical_analysis.get_bootstrap_interval_for_list_of_samples import \
get_bootstrap_interval_for_list_of_samples as get_bootstrap

def get_PCA_bootstraps(CC_dict):
    
    '''Given a list of param values for all trials for a pair of cells,
    determine the 98.3% confidence interval for that set of values via
    the bootstrapping method.  This will be used to determine significance
    of value for a given epoch relative to param = 0, and for significance
    of change in param from one epoch to another.'''    
    
    num_pairs = len(CC_dict['ongoing'])
    ong_CCs_all_pairs_all_trials = CC_dict['ongoing']    
    trans_CCs_all_pairs_all_trials = CC_dict['transient']    
    ss_CCs_all_pairs_all_trials = CC_dict['steady-state']    

    expl_var_boots = [] #1x3, of form [[upper, lower]_PC1, ... , [upper, lower]_PC3]
    pca_comps_boots = [] #3x3, where each row in each matrix contains the
                         #upper and lower conf limits of coeffs for ong, trans, and ss for that PC

    random_dist_var = [] #each entry is 1x3, of form [expl_var_PC1, ... , expl_var_PC3]
    random_dist_comps = [] #each entry is 3x3, where each row in each matrix contains the
                         #coeffs for ong, trans, and ss for that PC
    
    shuff_num = 0
    while shuff_num < 1000:
        resamp_avg_ong_CCs = [] #35 vals from resampling
        resamp_avg_trans_CCs = []
        resamp_avg_ss_CCs = []
        for k, val in enumerate(ong_CCs_all_pairs_all_trials): #index over pairs
            ong_CCs_all_trials = ong_CCs_all_pairs_all_trials[k]
            trans_CCs_all_trials = trans_CCs_all_pairs_all_trials[k]
            ss_CCs_all_trials = ss_CCs_all_pairs_all_trials[k]
            
            #re-sample CC vals for kth pair
            resamp_ong_CCs = []
            resamp_trans_CCs = []
            resamp_ss_CCs = []            
            while len(resamp_ong_CCs) <= len(ong_CCs_all_trials):
                random_ndx = random.randint(0, len(ong_CCs_all_trials) - 1)
                resamp_ong_CCs.append(ong_CCs_all_trials[random_ndx])
                resamp_trans_CCs.append(trans_CCs_all_trials[random_ndx])
                resamp_ss_CCs.append(ss_CCs_all_trials[random_ndx])
            resamp_avg_ong_CCs.append(np.mean(resamp_ong_CCs))
            resamp_avg_trans_CCs.append(np.mean(resamp_trans_CCs))
            resamp_avg_ss_CCs.append(np.mean(resamp_ss_CCs))
        
        #standardize the data        
        resamp_avg_ong_CCs = (resamp_avg_ong_CCs - np.mean(resamp_avg_ong_CCs))/np.std(resamp_avg_ong_CCs)
        resamp_avg_trans_CCs = (resamp_avg_trans_CCs - np.mean(resamp_avg_trans_CCs))/np.std(resamp_avg_trans_CCs)
        resamp_avg_ss_CCs = (resamp_avg_ss_CCs - np.mean(resamp_avg_ss_CCs))/np.std(resamp_avg_ss_CCs)
        
        #do PCA on re-sampled CC data
        CC_array = np.array([resamp_avg_ong_CCs, resamp_avg_trans_CCs, 
            resamp_avg_ss_CCs])
        CC_array = CC_array.transpose()
        pca = PCA(n_components = 3)
        
        pca.fit(CC_array)
        expl_var = pca.explained_variance_ratio_ #1x3 (1 X PC)
        pca_comps = pca.components_
        
        random_dist_var.append(expl_var)
        random_dist_comps.append(abs(pca_comps)) #3x3 (PC x epoch)        
        
        shuff_num += 1
    
    #get 98.3% confidence interval for each entry        
    for j, PC in enumerate(random_dist_var[0]): #j indexes over PCs
        boots = []
        all_vals = []
        for entry in random_dist_var: #cycle over 1000 entries (each 1 X 3)
            all_vals.append(entry[j]) #get jth-PC val for each of 1000 entries
        all_vals = sorted(all_vals)

        for k, val in enumerate(all_vals):
            if (k + 1) == int((0.05/(3*2.))*len(all_vals)):
                boots.append(val)
            elif (k + 1) == int((1-0.05/(3*2.))*len(all_vals)):
                boots.append(val)
        expl_var_boots.append(boots)
        
    for n, PC_index in enumerate(random_dist_comps[0]): #n indexes over PC number
        boots_for_PC = [] #1x3, where each intry is [upper, lower]
        for k, epoch in enumerate(random_dist_comps[0][n]): #k indexes over epochs
            boots_for_epoch = []
            coeffs_for_epoch = []
            for entry in random_dist_comps: #get each of the 1000 3x3 entries
                PC_coeffs = entry[n]/sum(entry[n]) #1x3 matrix of coeffs
                coeffs_for_epoch.append(PC_coeffs[k])
            coeffs_for_epoch = sorted(coeffs_for_epoch)

            for g, val in enumerate(coeffs_for_epoch):
                if (g + 1) == int((0.05/(3*2.))*len(coeffs_for_epoch)):
                    boots_for_epoch.append(val)
                elif (g + 1) == int((1-0.05/(3*2.))*len(coeffs_for_epoch)):
                    boots_for_epoch.append(val)
            boots_for_PC.append(boots_for_epoch)            
        pca_comps_boots.append(boots_for_PC)


    random_dist_var = []
    random_dist_comps = []
    
    return expl_var_boots, pca_comps_boots


def plot_CC_PCA_results(ax1, ax2,
    fontproperties, stim_types = ['extended_stim', 'flash'],
    windows = [2000.0, 400.0, 2000.0], gaps = [0.0, 200.0, 200.0],
    padding = 0.0, crit_freq = (20.0, 100.0), filt_kind = 'band',
    replace_spikes = True, remove_sine_waves = True,
    get_new_CC_results_all_pairs = False,
    get_new_processed_traces = False, 
    master_folder_path = 'E:\\correlated_variability'):

    epochs = ['ongoing', 'transient', 'steady-state']
        
    fonts = fontproperties
    size = fonts.get_size()

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
    
    ### get CCs by pair, all epochs and trials ###   
    CC_dict_by_pair_all_trials = get_CC_all_pairs(stim_types,
        windows, gaps, padding, crit_freq, filt_kind, replace_spikes, 
        remove_sine_waves, get_new_CC_results_all_pairs,
        get_new_processed_traces, master_folder_path) 
    
    CCs_all_trials_by_pair_and_epoch = {} #re-format the above dict for PCA
    for epoch in epochs:
        CCs_all_trials_by_pair_and_epoch[epoch] = []
        for pair in CC_dict_by_pair_all_trials:
            CCs_by_epoch_for_pair = CC_dict_by_pair_all_trials[pair]
            CCs_all_trials_for_pair_this_epoch = CCs_by_epoch_for_pair[epoch]
            CCs_all_trials_by_pair_and_epoch[epoch].append(CCs_all_trials_for_pair_this_epoch)
        
        

    CC_dict = {}
    bootstrap_ranges = {}
    
    for epoch in epochs:
        CC_dict[epoch] = []
        bootstrap_ranges[epoch] = []               

    ### populate the dictionaries ###

    
    #CC_dict_by_pair_all_trials = {pair_name:{'ongoing':[CC_trial1, 
    #    ..., CC_trialn], ..., 'steady-state':[CC_trial1,
    #    ..., CC_trialn]]}}

    for pair_name in CC_dict_by_pair_all_trials:
        CC_dict_for_cell = CC_dict_by_pair_all_trials[pair_name]

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

        #pickle.dump(master_dict, open(pop_CC_dict_path, 'wb'))
    
    #standardize the variables (makes a HUGE difference here)
    ong_CCs = CC_dict['ongoing']
    trans_CCs = CC_dict['transient']
    ss_CCs = CC_dict['steady-state']
    ong_CCs = (ong_CCs - np.mean(ong_CCs))/np.std(ong_CCs)
    trans_CCs = (trans_CCs - np.mean(trans_CCs))/np.std(trans_CCs)
    ss_CCs = (ss_CCs - np.mean(ss_CCs))/np.std(ss_CCs)
    
    
    ### do PCA on data, and get confidence intervals ###
    CC_array = np.array([ong_CCs, trans_CCs, ss_CCs])
    CC_array = CC_array.transpose()
    pca = PCA(n_components = 3)
    pca.fit(CC_array)
    expl_var = pca.explained_variance_ratio_
    pca_comps = pca.components_
    expl_var_boots, pca_comps_boots = get_PCA_bootstraps(CCs_all_trials_by_pair_and_epoch)
    

    ### plot explained variance by PC ###
    xs = [0, 1, 2]
    lowers = []
    uppers = []
    for j, entry in enumerate(expl_var): #j indexes over PC
        lower = expl_var[j] - expl_var_boots[j][0]
        upper = expl_var_boots[j][1] - expl_var[j]
        lowers.append(lower)
        uppers.append(upper)
    yerrs = [lowers, uppers]
    ax1.errorbar(xs, expl_var, yerr = yerrs, fmt = '--o', color = 'k',
        elinewidth = 1.0, markersize = 4)
    ax1.set_xlim(-0.1, 2.1)
    x_labels = ['', 'PC1', 'PC2', 'PC3']
    loc = plticker.MultipleLocator(base = 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xticklabels(x_labels, fontsize = size)
    ax1.xaxis.set_major_locator(loc)
    ax1.set_ylim(0, 0.62)
    locy = plticker.MultipleLocator(base = 0.3)
    ax1.yaxis.set_major_locator(locy)
    y_labels = ax1.get_yticks()
    ax1.set_yticklabels(y_labels, fontsize = size)
    
    ax1.set_ylabel('% variance explained', fontsize = size)
    ax1.tick_params('both', length = 3, width = 1.0, which = 'major',
        direction = 'outward')
    
    #print results to terminal
    print ''
    print '  variance explained by each PC'
    for k, entry in enumerate(expl_var): #k indexes over PC
        var_label =  '    PC%s:'%str(k+1)
        var_label += str(100*round(expl_var[k], 3)) + '%'
        var_label += ' (%s - '%str(100*round(expl_var_boots[k][0], 3))
        var_label += '%s'%str(100*round(expl_var_boots[k][1], 3)) + '%)'
        print var_label
        
    ### plot the variance explained by each PC for each epoch (bar graph) ###
    #index = np.arange(3) + 0.1
    index = [1, 2, 3]
    #index = pylab.array([0, 1.5, 2.5])
    bar_width = 0.5
    opacity = 0.5
    
    PC1_boots = pca_comps_boots[0]
    
    PC1_comp_norm = abs(pca_comps[0][0]) + abs(pca_comps[1][0]) + abs(pca_comps[2][0])
    
    PC1_yerrs = [[],[]] #2[lower, upper] X 3epochs

    
    for j, entry in enumerate(PC1_boots): #j is epoch index
        PC1_boots_for_epoch = PC1_boots[j]
        PC1_comp = abs(pca_comps[0][j])/PC1_comp_norm    
        PC1_lower_for_epoch = PC1_comp - PC1_boots_for_epoch[0]
        PC1_upper_for_epoch= PC1_boots_for_epoch[1] - PC1_comp
        PC1_yerrs[0].append(PC1_lower_for_epoch)
        PC1_yerrs[1].append(PC1_upper_for_epoch)
    
                      
    expl_var_PC1 = abs(pca_comps[0])/PC1_comp_norm

    #print results to terminal    
    print ''
    print '  variance explained by PC1 in each epoch'
    for n, entry in enumerate(expl_var_PC1):
        var_label = '    %s:'%epochs[n]
        var_label += ' ' + str(100*round(expl_var_PC1[n], 3)) + '%'
        var_label += ' (' + str(100*round(expl_var_PC1[n] - PC1_yerrs[0][n], 3))
        var_label += ' - ' + str(100*round(expl_var_PC1[n] + PC1_yerrs[1][n], 3)) + '%)'
        print var_label
    
    rects = ax2.bar(index, expl_var_PC1, bar_width,
                     alpha=opacity, align = 'center',
                     color=['y', 'b', 'g'],
                     ecolor = 'k',
                     yerr = PC1_yerrs)
    

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('left')
    #ax.set_yticklabels([])
    locy = plticker.MultipleLocator(base = 0.3)
    locx = plticker.MultipleLocator(base = 1.0)
    x_labels = ['', 'ong', 'trans', 'ss']
    ax2.set_xticklabels(x_labels, fontsize = size)
    
    ax2.yaxis.set_major_locator(locy)
    ax2.xaxis.set_major_locator(locx)

    ax2.hlines((1./3.), 0.5, 3.5, color = 'r', lw = 1.0, linestyles = '--')
    ax2.set_ylim(0, 0.6)
    ax2.set_xlim(0.45, 3.55)
    y_labels = ax2.get_yticks()
    ax2.set_yticklabels(y_labels, fontsize = size)
    ax2.text(0.5, 0.9, 'PC1 only', fontsize = size, 
        horizontalalignment = 'center', transform = ax2.transAxes)
    
    ax2.set_ylabel('% variance explained', fontsize = size)
    ax2.tick_params('both', length = 3, width = 1.0, which = 'major',
        direction = 'outward')    

