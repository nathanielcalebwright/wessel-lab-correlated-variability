'''
copyright (c) 2016 Nathaniel Wright

this program is free software: you can redistribute it and/or modify
it under the terms of the gnu general public license as published by
the free software foundation, either version 3 of the license, or
(at your option) any later version. you should have received a copy
of the gnu general public license along with this program.  if not,
see <http://www.gnu.org/licenses/>.
    
'''

from scipy.optimize import leastsq
import numpy


def make_sine_wave(num_samples, sine_freq, samp_freq, phase, amplitude):
    t = numpy.arange(num_samples, dtype=numpy.float64)/samp_freq
    w = 2.0*numpy.pi*(sine_freq)
    return amplitude*numpy.sin(w*t - phase)

def fit_sine_wave(trace, window_for_fit_ms = (0, 1000.0), sine_freq = 60.0,
    samp_freq = 10000.0, warn = False):
    start_ndx = window_for_fit_ms[0]*samp_freq/1000.0
    stop_ndx = window_for_fit_ms[1]*samp_freq/1000.0
    trace_for_fit = trace[start_ndx:stop_ndx]

    f = lambda x: trace_for_fit - make_sine_wave(len(trace_for_fit), sine_freq,
        samp_freq, x[0], x[1])
    results = leastsq(f, (0.0, 30.0), full_output=warn, 
            maxfev=10000)
    phase, amplitude = results[0]

    full_sine_wave = make_sine_wave(len(trace), sine_freq, samp_freq, phase,
        amplitude)

    return full_sine_wave

def remove_sine_waves(trace, window_for_fit_ms = (0, 1000.0), sine_freqs = [60],
        samp_freq = 1000.0):

    '''
    Given a single voltage trace, fit a sine wave to the 
    first window_for_fit_ms, and subtract a 'line noise' signal with
    resulting amplitdue and phase from the full trace.
    
    Parameters
    ---------- 
    trace : pylab array
        full voltage trace
    window_for_fit_ms : tuple of floats
        section of full trace to use for sine wave fit
    sine_freqs : python list of ints or floats
        list of 'line noise' frequencies to fit and remove
    samp_freq: float
        sampling frequency of the trace (Hz)

    Returns
    -------
    trace: numpy array
        full trace, with sine wave(s) removed
    
    '''
    
    for sine_freq in sine_freqs:
        sine_wave = fit_sine_wave(trace, window_for_fit_ms,
            sine_freq, samp_freq)
        trace += -1*sine_wave

    return trace
