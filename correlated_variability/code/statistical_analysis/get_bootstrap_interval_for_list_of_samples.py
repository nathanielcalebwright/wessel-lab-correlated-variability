import pylab
from numpy import random

def get_bootstrap_interval_for_list_of_samples(list_of_samples, 
        num_reps = 1000, conf_level = 0.95, num_comparisons = 1):
    
    '''Given a list of samples, determine the desired confidence interval
    (defined by conf_level*100%) by bootstrapping (i.e., sampling with
    random replacement).  Correct the confidence limits for number of
    simultaneous comparisons (num_comparisons).
    
    Parameters
    ----------
    list_of_samples : python list
        list of values calculated from the data (usually single-trial values
        for one cell/pair, or across-trial averages for a population of 
        cells/pairs)
    num_reps: int
        number of times to repeat the re-sampling process
    conf_level: float
        confidence level of the bootstrap bands, BEFORE correcting for
        multiple comparisons
    num_comparisons : int
        number of simultaneous comparisons to be made with the resulting data;
        will be used to adjust the confidence level
    
    '''    
    
    bootstrap = []
    alpha_uncorrected = 1.0-conf_level #0.05

    random_distribution_means = []
    shuff_num = 0
    while shuff_num <= num_reps:
        random_dist = []
        while len(random_dist) <= len(list_of_samples):
            random_ndx = random.randint(0, len(list_of_samples) - 1)
            random_dist.append(list_of_samples[random_ndx])
        random_distribution_means.append(pylab.mean(random_dist))
        shuff_num += 1
    random_distribution_means = sorted(random_distribution_means)
    for k, val in enumerate(random_distribution_means):
        if (k + 1) == int((alpha_uncorrected/(float(2*num_comparisons)))*len(random_distribution_means)):
            bootstrap.append(val)
        elif (k + 1) == int((1 - (alpha_uncorrected/(float(2*num_comparisons))))*len(random_distribution_means)):
            bootstrap.append(val)               

    return bootstrap