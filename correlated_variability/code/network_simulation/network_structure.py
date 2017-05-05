
"""
copyright (c) 2016 Mahmood Hoseini

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.

"""

import numpy as np

def MatMaker (n1, n2, prob) :
    A = np.random.rand(n1, n2)
    A[A > prob] = 0;
           
    return A


def get_random_network (N_I, N_E, pii, pie, pei, pee) :
    
    W_II = MatMaker(N_I, N_I, pii); np.fill_diagonal(W_II, 0)
    W_EE = MatMaker(N_E, N_E, pee); np.fill_diagonal(W_EE, 0)
    W_EI = MatMaker(N_E, N_I, pei)        
    W_IE = MatMaker(N_I, N_E, pie)
    
    return W_II, W_EE, W_IE, W_EI

        
def get_ring_net(rows, cols, num_conn) :
    ''' This functions makes a ring-like network, where nonzero connection
    weights are drawn from a continuous uniform distribution.'''
    W = np.zeros((rows, cols))
    vec = np.zeros(cols)
    vec[1:num_conn/2 + 1] = 2
    vec[cols-num_conn/2 : cols] = 2
    for ii in range(rows) :
        W[ii , :] = vec*np.random.random(cols)
        vec = np.roll(vec, 1)
    return W

def get_ring_net_beta(rows, cols, num_conn) :
    ''' This functions makes a ring-like network, where nonzero connection
    weights are drawn from a beta distribution.'''
    W = np.zeros((rows, cols))
    vec = np.zeros(cols)
    vec[1:num_conn/2 + 1] = 10
    vec[cols-num_conn/2 : cols] = 10
    for ii in range(rows) :
        W[ii , :] = vec*np.random.beta(0.11, 1, cols)
        vec = np.roll(vec, 1)
    return W


def get_clustered_forward_backward_network(N_I, N_E, k_I, k_E, 
    p_rw, exc_syn_weight_dist = 'beta') :
        
    '''This functions recieves number of inh and exc neurons, along with number
       of inh/exc inputs to each individual neuron (k_I/k_E) and also rewiring
       probability. Returns the synaptic weight matrices corresponding to a 
       clustered network.  For each nonzero connection, the weight is drawn 
       from either a continuous uniform or beta distribution.
       Inputs :
               N_I : number of interneurons
               N_E : number of pyramidals
               p_rw: rewiring probability
       Outputs :
               Four synaptic weight matrices each for one type of connection 
    '''
    if exc_syn_weight_dist == 'beta':    
        W_II = get_ring_net_beta(N_I, N_I, k_I)
        
        W_EE = get_ring_net_beta(N_E, N_E, k_E)
    else:
        W_II = get_ring_net(N_I, N_I, k_I)
        
        W_EE = get_ring_net(N_E, N_E, k_E)
        
    W_IE = np.zeros((N_I, N_E))
    vec = np.zeros(N_I)
    vec[:k_E/2] = 2; vec[-k_E/2:] = 2
    for i in range(N_E) :
        W_IE[:, i] = vec*np.random.random(N_I)
        if (i+1)%4 == 0 :
            vec = np.roll(vec, 1)
    W_IE = np.roll(W_IE, 2, axis=1);
    
    W_EI = np.zeros((N_E, N_I))
    vec = np.zeros(N_E)
    vec[:k_I/2+2] = 2; vec[-k_I/2+2:] = 2
    for i in range(N_I) :
        W_EI[:, i] = vec*np.random.random(N_E)
        vec = np.roll(vec, 4)
    
    for i in range(N_I) :
        for j in range(N_I) :
            if (W_II[i, j] != 0 and np.random.rand() < p_rw) :
                    W_II[i, j] = 0
                    Bool = True
                    while (Bool) :
                        m = np.random.randint(0, N_I)
                        if W_II[i, m] == 0 and m != i :
                            W_II[i, m] = 10*np.random.beta(0.11, 1)
                            Bool = False
        for j in range(N_E) :
            if (W_IE[i, j] != 0 and np.random.rand() < p_rw) :
                    W_IE[i, j] = 0
                    Bool = True
                    while (Bool) :
                        m = np.random.randint(0, N_E)
                        if W_IE[i, m] == 0 and m != i :
                            W_IE[i, m] = 10*np.random.beta(0.11, 1)
                            Bool = False
                            
    for i in range(N_E) :
        for j in range(N_E) :
            if (W_EE[i, j] != 0 and np.random.rand() < p_rw) :
                    W_EE[i, j] = 0
                    Bool = True
                    while (Bool) :
                        m = np.random.randint(0, N_E)
                        if W_EE[i, m] == 0 and m != i :
                            W_EE[i, m] = 10*np.random.beta(0.11, 1)
                            Bool = False
        for j in range(N_I) :
            if (W_EI[i, j] != 0 and np.random.rand() < p_rw) :
                    W_EI[i, j] = 0
                    Bool = True
                    while (Bool) :
                        m = np.random.randint(0, N_I)
                        if W_EI[i, m] == 0 and m != i :
                            W_EI[i, m] = 10*np.random.beta(0.11, 1)
                            Bool = False
                            
    return W_II, W_EE, W_IE, W_EI
