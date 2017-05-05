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

def get_possible_pairs(list_of_cells):    
    if len(list_of_cells) == 2:
        possible_pairs = [list_of_cells]
    else:
        possible_pairs = []
        possible_pairs.append([list_of_cells[0], list_of_cells[1]])
        possible_pairs.append([list_of_cells[0], list_of_cells[2]])
        possible_pairs.append([list_of_cells[1], list_of_cells[2]])
    return possible_pairs

def check_chosen_pair(stim_type, cells_to_plot, master_folder_path):
    '''Make sure the manually-entered pair was actually a simultaneously-
    recorded pair.  If it is, things will go smoothly.  If not, the 
    corresponding panel will be empty, and this function will print a 
    list of valid cell choices for the given stimulus type.
    
    Parameters
    ----------
    stim_type: string
        'extended_stim' or 'flash'
    cells_to_plot: list of strings
        putative list of simultaneously-recorded cells from recording session
        (e.g., ['070314_c2', '070314_c3'])
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    plot_decision: bool
        True if selected pair is valid

    '''
    
    if stim_type == 'extended_stim':
        stim_names = ['naturalistic movie',
            'motion-enhanced movie', 'phase-shuffled motion-enhanced movie', 
            'gray screen']
    else:
        stim_names = ['sub-field red flash', 'whole-field red flash']    
    
    cell_groups_by_stim_path = master_folder_path + '\\misc\\cell_groups_by_stimulus_type.p'
    cell_groups_by_stim = pickle.load(open(cell_groups_by_stim_path, 'rb'))

    possible_pairs_for_stim_type = []
    for stim_name in stim_names:
        cell_groups_for_stim = cell_groups_by_stim[stim_name]
        for cell_group in cell_groups_for_stim:
            pair_list = get_possible_pairs(cell_group)
            possible_pairs_for_stim_type.extend(pair_list)
    if cells_to_plot in possible_pairs_for_stim_type:
        plot_decision = True
    else:
        plot_decision = False            
    if plot_decision == False:
        print cells_to_plot, 'not a valid pair choice for %s.'%stim_type
        print 'corresponding panel will be blank.'
        print 'instead, please choose from:'
        print possible_pairs_for_stim_type
    return plot_decision