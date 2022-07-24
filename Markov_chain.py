import numpy as np
import math
from numpy.core.records import array
from Alias import alias 
    # A is the transition matrix for the Markov chain
    # T is the number of time steps
    # The columns of z0 are the initial states for the realizations

def markov_chain(T, A, y0):

    # A should be square, each column is a p.m.f.
    d = np.size(A,1)
    # convert transition matrix to samplers
    a = []
    for k in range(0,d):
        a.append(alias(A[:,k]))

    nreal = len(y0)
    y = np.full((T,nreal), np.nan)
    y[0,:] = np.transpose(y0)

    for t in range(1,T):
        for j in range(1,d+1):
            isj = np.where(y[t-1,:] == j)[0]
            # print(len(isj))
            y[t, isj] = a[j-1](len(isj)) + 1
    # print(y)
    return y
