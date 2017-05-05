"""
Copyright (C) 2011  Nathaniel Wright

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

def get_spike_indices_from_slope(trace, samp_freq, thresh):
    spike_indices = []
    slope_trace = []
    k = 1
    while k < len(trace):
        slope_trace.append(trace[k] - trace[k-1])
        k += 1
    for index, val in enumerate(slope_trace[:-150]):
        if val >= 3 and val >= slope_trace[index - 1] and val >= slope_trace[index + 1]:
            #find the next zero of the slope; that's the AP peak
            j = index + 1
            while j < len(slope_trace[:-150]):
                #print slope_trace[j]
                if slope_trace[j] <= 0:
                    #print j
                    spike_indices.append(j)
                    break
                else:
                    j += 1
                    
    return spike_indices


def replace_all_spikes(Vm, samp_freq = 1000.0, thresh = 15,
        removal_window = 50):
    ''' 
    Replace a window from a single Vm recording centered on each spike,
    with a line connecting the membrane potential values at the two ends
    of the window. Necessary for correlation and coherence of subthreshold
    activity.
    
    Parameters
    ----------
    Vm : numpy array
        voltage trace (unfiltered, spikes not removed)
    samp_freq : float
        sampling frequency (Hz)
    thresh: int or float
        threshold used to detect spikes
    removal_window: int or float
        size of window (in ms, centered on AP peak) to be replaced w/straight
        line (fit via interpolation)
    
    Returns
    -------
    Vm_for_analysis: numpy array
        voltage trace (unfiltered, spikes removed)
    
    '''
        
    Vm_for_analysis = Vm
    spike_indices1 = get_spike_indices_from_slope(Vm, samp_freq, thresh)
    #spike_indices1 = get_spike_indices(Vm, samp_freq, thresh)
    if len(spike_indices1) > 0:
        b = -1 #need to start with last spike in trace
        while abs(b) <= len(spike_indices1):
            spike_index = spike_indices1[b]
            #print spike_index
            trace_index = spike_index - int(0.5*removal_window*samp_freq/1000.0)
            
            #get linear fit between two points separated by removal_window
            #ms, centered on spike_index
            start_index = spike_index - int(0.5*removal_window*samp_freq/1000.0)
            stop_index = start_index + int(removal_window*samp_freq/1000.0)
            start_val = Vm_for_analysis[start_index]
            
            stop_val = Vm_for_analysis[stop_index]
            lin_fit = [start_val]
            j = 1
            while j <= (stop_index - start_index):
                lin_fit.append(lin_fit[j-1] - (start_val - stop_val)/int(stop_index - start_index))
                j += 1
            
            for k, val in enumerate(lin_fit):
                Vm_for_analysis[k + trace_index] = lin_fit[k]

            b += -1

    return Vm_for_analysis

            

