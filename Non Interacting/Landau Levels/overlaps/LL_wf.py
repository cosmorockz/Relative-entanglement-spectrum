import numpy as np
from mpmath import *

mp.dps = 30


def SLL_normalization(Q, l, m):
    # Normalization constant for the Spherical Landau Level
    # Only depends on the charge Q, Landau level l and angular momentum m
    return sqrt(((2*l+1)/(4*pi)) * binomial(2*l, l-Q)/binomial(2*l, l-m))


def SLL(theta, phi, Q, l, m):
    # Spherical Landau Level
    # Q is the charge, l is the Landau level, m is the angular momentum
    # this function outputs the value (complex number) of the SLL wavefunction at a specified angular coordinate
    u = cos(theta/2)  # * np.exp(1j*phi/2)
    v = sin(theta/2)  # * np.exp(-1j*phi/2)
    pre = SLL_normalization(Q, l, m) * (-1)**(l-m) * \
        v**(Q-m) * u**(Q+m)  # part before the summation
    # part inside the summation
    sum_part = 0
    for s in range(0, np.int(l-m+1)):
        if (l-Q >= s) and (l+Q >= l-m-s):
            sum_part += (-1)**s * binomial(l-Q, s) * binomial(l+Q, l-m-s) * \
                (v*v)**(l-Q-s) * (u*u)**s
    wf = pre * sum_part  # total wavefunction
    return wf