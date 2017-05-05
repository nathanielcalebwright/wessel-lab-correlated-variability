import os
import cPickle as pickle
import numpy
import pylab

def downsample_trace(trace, ndx_skips):
    downsamp_trace = []
    i = 0
    while i < len(trace):
        downsamp_trace.append(pylab.mean(trace[i:(i+ndx_skips)]))

        i += ndx_skips
    downsamp_trace = pylab.array(downsamp_trace)
    return downsamp_trace

def get_downsampled_model_LFPs(exc_connectivity = 'clustered', 
    net_state = 'synchronous', int_SD = True, num_trials = 15,
    master_folder_path = 'F:\\correlated_variability_'):
    
    ### first, see if saved dictionary of downsamp LFPs exists ###    
    save_path = master_folder_path +  '\\model_results\\'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path += exc_connectivity 
    save_path += '_' + net_state
    if int_SD == False:
        save_path += '_no_SD'
    save_path += '\\LFPs\\'
    if not os.path.isdir(save_path):
        os.path.mkdir(save_path)
    LFP_save_path = save_path + 'downsamp_LFPs_trial%s.p'%str(num_trials)
    if os.path.isfile(LFP_save_path):
        LFP_dict = pickle.load(open(LFP_save_path, 'rb'))
    else:        
        #get new dict if necessary
        sim_samp_freq = 20000.0   
        
        ### access array of saved simulated LFPs ###
        array_path = save_path + 'LFPs_trial%s.npy'%str(num_trials)
        LFP_array = numpy.load(array_path)
        num_LFPs = len(LFP_array)
        
        ### downsample each LFP, save to dictionary ###
        LFP_dict = {}
        LFP_num = 1
        while LFP_num <= num_LFPs:
            LFP = LFP_array[LFP_num - 1]
            #downsample the trace
            ndx_skips = int(sim_samp_freq/1000.0)
            LFP = downsample_trace(LFP, ndx_skips)
            
            #add to dictionary
            LFP_dict['LFP_%s'%str(LFP_num)] = LFP
            LFP_num += 1
    
        pickle.dump(LFP_dict, open(LFP_save_path, 'wb'))
    
    return LFP_dict
    
    