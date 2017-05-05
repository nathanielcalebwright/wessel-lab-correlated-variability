"""
copyright (c) 2015 Mahmood Hoseini

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.

"""
import numpy as np

def auxiliary(N, ntsteps, rate, dt) :
         
    spk_train = np.random.random((N, ntsteps))
    spk_train[spk_train <= rate*dt] = 1;
    spk_train[(spk_train > rate*dt) & (spk_train != 1)] = 0;
    
    return spk_train


def get_ext_input(N_I, N_E, r_I, r_E, stim_factor, t_max, t_stim, dt) :
    """ Step function external input."""
    ext_input_I = np.concatenate([auxiliary(N_I, t_stim/dt, r_I, dt*1e-3), 
                                  auxiliary(N_I, (t_max-t_stim)/dt, r_I*stim_factor, dt*1e-3)], axis=1)
    ext_input_E = np.concatenate([auxiliary(N_E, t_stim/dt, r_E, dt*1e-3), 
                                  auxiliary(N_E, (t_max-t_stim)/dt, r_E*stim_factor, dt*1e-3)], axis=1)
        
    return ext_input_I, ext_input_E

def get_ext_input_linear_increase(N_I, N_E, r_I, r_E, stim_factor, t_max, 
        t_stim, time_to_max, dt) :
    """ Linear increase in external input."""
    
    ext_input_I = auxiliary(N_I, t_stim/dt, r_I, dt*1e-3)
    ext_input_E = auxiliary(N_E, t_stim/dt, r_I, dt*1e-3)
    
    t = t_stim
    total_steps = time_to_max/50 #increase to max in 50ms increments
    step_num = 1
    while t < t_stim + time_to_max:
        r_I_new = r_I*(1 + (float(step_num)/total_steps)*(stim_factor - 1))
        r_E_new = r_E*(1 + (float(step_num)/total_steps)*(stim_factor - 1))   

        I_train_for_window = auxiliary(N_I, 50/dt, r_I_new, dt*1e-3)
        E_train_for_window = auxiliary(N_E, 50/dt, r_E_new, dt*1e-3)
        
        ext_input_I = np.concatenate((ext_input_I, I_train_for_window), axis = 1)
        ext_input_E = np.concatenate((ext_input_E, E_train_for_window), axis = 1)
               
        t += 50 #ms
        step_num += 1
    
    remaining_I_train = auxiliary(N_I, (t_max-t_stim - time_to_max)/dt, 
        r_I*stim_factor, dt*1e-3)
    remaining_E_train = auxiliary(N_E, (t_max-t_stim - time_to_max)/dt, 
        r_E*stim_factor, dt*1e-3)
    
    ext_input_I = np.concatenate((ext_input_I, remaining_I_train), axis = 1)
    ext_input_E = np.concatenate((ext_input_E, remaining_E_train), axis = 1)
    
    return ext_input_I, ext_input_E
    

