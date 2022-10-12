import numpy as np
from mpmath import *
import csv

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

def overlap_integrand(theta, Q, l1, l2, m):
    wf1 = SLL(theta,0,Q,l1,m)
    wf2 = SLL(theta,0,Q,l2,m)
    x = 2 * pi * sin(theta) *( wf1 * wf2 )
    return x

def overlap_integral(Q, l1, l2, m, theta_cut):
    x = quad(lambda theta: overlap_integrand(
        theta, Q, l1, l2, m), [0, theta_cut])
    return x

def OI_file_generator(Q):
    l_max = Q+10
    theta_cut = pi/2
    M = [-l_max+i for i in range(2*l_max+1)]
    data = {}
    for m in M:
        n_ll = l_max - Q + 1
        N = l_max - np.abs(m) + 1
        if N >= n_ll:
            for i in range(n_ll):
                for j in range(n_ll):
                    key = str(Q+i)+str(',')+str(Q+j)+str(',')+str(m)
                    value = overlap_integral(Q, Q+i, Q+j, m, theta_cut)
                    data[key] = value
        elif N <= n_ll:
            for i in range(N):
                for j in range(N):
                    key = str(np.abs(m)+i)+str(',')+str(np.abs(m)+j)+str(',')+str(m)
                    value = overlap_integral(Q, np.abs(m)+i, np.abs(m)+j, m, theta_cut)
                    data[key] = value

    with open('LL_OI_Q_'+str(Q)+'.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in data.items():
           writer.writerow([key, value])
    return None

for i in range(1,21):
    OI_file_generator(i)
