'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''
import numpy as np

def plot_stim_trace(ax, windows, gaps, padding):
    '''
    Create a 'binary' trace indicating stim onset and duration.
    
    Parameters
    ----------
    ax: matplotlib axis
    windows: list of floats
        widths of ongoing, transient, and steady-state windows (ms)
    gaps: list of floats
        sizes of gaps between stim onset and end of ongoing, stim onset and
        beginning of transient, end of transient and beginning of 
        steady-state (ms)
    padding: list of floats
        size of window (ms) to be added to the beginning and end of each 
        subtrace (only nonzero when doing wavelet filtering)
    extended_stim_cells_to_plot : list of strings
        list of simultaneously-recorded cells to plot in A (extended stim)
        (e.g., ['070314_c2', '070314_c3'])
    
    Returns
    -------
    none   
    '''

    stim_trace_len = int(sum(windows) + sum(gaps) + 2*padding)
    stim_trace = np.zeros(stim_trace_len)
    stim_trace[int(padding + windows[0] + gaps[0]):]+= 5
    ax.plot(stim_trace, 'm', lw = 1.5)
    ax.set_ylim(-1, 7)
    ### turn off axes and tick marks ###    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)