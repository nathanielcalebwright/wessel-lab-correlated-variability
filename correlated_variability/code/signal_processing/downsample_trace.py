import pylab

def downsample_trace(trace, ndx_skips = 10):
    new_trace = []
    i = 0
    while i < len(trace):
        new_trace.append(pylab.mean(trace[i:(i+ndx_skips)]))

        i += ndx_skips
    return pylab.array(new_trace)
