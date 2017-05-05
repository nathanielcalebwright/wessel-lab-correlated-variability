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

def get_cell_pairs_by_stim_type(stim_types, master_folder_path):
    '''
    For stimulus types of interest (e.g., 'extended_stim' and/or 'flash'),
    get all cell pairs used to generate results in the manuscript, segregated 
    by stim_type.
    
    Parameters
    ----------
    stim_type: string
        'extended_stim' or 'flash'
    master_folder_path: string
        full path of directory containing data, code, figures, etc.
        
    Returns
    -------
    cell_pairs_by_stim_type: python dictionary
        dictionary of subset of cell pairs used to generate results in 
        manuscript, where keys are stim type (e.g., 'extended_stim' and/or 
        'flash')

    '''
    
    cell_pairs_by_stim_type = {}
    
    extended_stim_names = ['naturalistic movie',
            'phase-shuffled motion-enhanced movie', 'motion-enhanced movie',
            'gray screen']
    
    flash_names = ['sub-field red flash', 'whole-field red flash']   

    
    cell_groups_by_stim_path = master_folder_path + '\\misc\\cell_groups_by_stimulus_type.p'
    cell_groups_by_stim = pickle.load(open(cell_groups_by_stim_path, 'rb'))

    
    for stim_type in stim_types:
        cell_pairs_by_stim_type[stim_type] = []
        #get stim_names associated w/this stim_type
        if 'flash' in stim_type:
            stim_names_wanted = flash_names
        else:
            stim_names_wanted = extended_stim_names
        #get all cell groups for each stim_name_wanted
        for stim_name_wanted in stim_names_wanted:
            cell_groups_wanted = cell_groups_by_stim[stim_name_wanted]
            #get all possible pairs for each cell_group_wanted
            possible_pairs_for_stim_name_wanted = []
            for cell_group_wanted in cell_groups_wanted:
                pair_list = get_possible_pairs(cell_group_wanted)
                possible_pairs_for_stim_name_wanted.extend(pair_list)
            cell_pairs_by_stim_type[stim_type].extend(possible_pairs_for_stim_name_wanted)
    
    return cell_pairs_by_stim_type