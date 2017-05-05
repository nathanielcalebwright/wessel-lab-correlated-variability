'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''


import cPickle as pickle

def get_possible_triples(list_of_trodes): 
    LFP_name = list_of_trodes[-1]
    if len(list_of_trodes) == 3:
        possible_triples = [list_of_trodes]
    else:
        possible_triples = []
        possible_triples.append([list_of_trodes[0], list_of_trodes[1], LFP_name])
        possible_triples.append([list_of_trodes[0], list_of_trodes[2], LFP_name])
        possible_triples.append([list_of_trodes[1], list_of_trodes[2], LFP_name])
    return possible_triples

def check_chosen_triple(stim_type, trodes_to_plot, master_folder_path):
    '''Make sure the manually-entered V-V-LFP triple was actually a simultaneously-
    recorded triple.  If it is, things will go smoothly.  If not, the 
    corresponding panel will be empty, and this function will print a 
    list of valid cell choices for the given stimulus type.
    
    Parameters
    ----------
    stim_type: string
        'extended_stim' or 'flash'
    trodes_to_plot: list of strings
        putative list of simultaneously-recorded trodes from recording session
        (e.g., ['070314_c2', '070314_c3', '070314_LFP2'])
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    plot_decision: bool
        True if selected triple is valid

    '''
    
    if stim_type == 'extended_stim':
        stim_names = ['naturalistic movie',
            'motion-enhanced movie', 'phase-shuffled motion-enhanced movie', 
            'gray screen']
    else:
        stim_names = ['sub-field red flash', 'whole-field red flash']    
    
    trode_groups_by_stim_path = master_folder_path + '\\misc\\trode_groups_by_stimulus_type.p'
    trode_groups_by_stim = pickle.load(open(trode_groups_by_stim_path, 'rb'))

    possible_triples_for_stim_type = []
    for stim_name in stim_names:
        trode_groups_for_stim = trode_groups_by_stim[stim_name]
        for trode_group in trode_groups_for_stim:
            LFP_in_group = False
            for trode_name in trode_group:
                if 'LFP' in trode_name:
                    LFP_in_group = True
                    break
            if LFP_in_group:
                triple_list = get_possible_triples(trode_group)
                possible_triples_for_stim_type.extend(triple_list)
    if trodes_to_plot in possible_triples_for_stim_type:
        plot_decision = True
    else:
        plot_decision = False            
    if plot_decision == False:
        print trodes_to_plot, 'not a valid choice for %s.'%stim_type
        print 'corresponding panel will be blank.'
        print 'instead, please choose from:'
        print possible_triples_for_stim_type
    return plot_decision