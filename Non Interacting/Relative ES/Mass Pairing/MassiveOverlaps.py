import numpy as np
from mpmath import *
import csv

mp.dps = 30
mz = 4
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

def dirac_wf(theta,phi,Q,n,la,m):
    # This is the correct wavefunction only when the magnetic monopole 
    # is 0, otherwise we will have to include a (sqrt(2))^(\delta_{n,0})
    e = sqrt(n**2 + mz**2)
    # Component 1
    if la == -1:
	    comp1 = (-1)**(1/2 + m) * sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
	    # Component 2
	    comp2 = (-1)**(1/2 + m) * sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
    if la == 1:
	    comp1 = sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
	    # Component 2
	    comp2 = sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
    wf = matrix([comp1,comp2])/sqrt(2*e)
    return wf

def overlap_integrand(theta, Q, n1, n2, la1, la2, m):
    wf1 = dirac_wf(theta,0,Q,n1,la1,m)
    wf2 = dirac_wf(theta,0,Q,n2,la2,m)
    x = 2 * pi * sin(theta) *( wf1[0] * wf2[0] + wf1[1] * wf2[1] )
    return x

def overlap_integral(Q, n1, n2, la1, la2, m, theta_cut):
    x = quad(lambda theta: overlap_integrand(
        theta, Q, n1, n2, la1, la2, m), [0, theta_cut])
    return x

def permitted_levels(n_cutoff,m):
    M = int(abs(m)+0.5)
    levels = []

    for i in range(n_cutoff,M-1,-1):
        levels.append(i)
    
    return levels

Q = 0
Q_down = Q - 0.5
Q_up = Q + 0.5
n_cutoff = 10

theta_cut = pi/2
l_max = Q_down+n_cutoff
mm = [-l_max+i for i in range(int(2*l_max+1))]

data1 = {}
data2 = {}
for m in mm:
    levels = permitted_levels(n_cutoff,m)
    N = len(levels)
    for i in range(N):
        for j in range(N):
            val1 = overlap_integral(Q, levels[i], levels[j], 1, 1, m, theta_cut)
            val2 = overlap_integral(Q, levels[i], levels[j], -1, -1, m, theta_cut)
            valP = 1/2 * (val1 + val2)
            valN = 1/2 * (val1 + val2)
            if abs(valP) < 1e-30:
                valP = 0
            if abs(valN) < 1e-30:
                valN = 0
            k = str(levels[i])+str(",")+str(levels[j])+ str(",")+str(m)
            data1[k] = valP
            data2[k] = valN

with open('SingleBandCutoff_'+str(n_cutoff)+'_M'+str(mz)+'P.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in data1.items():
       writer.writerow([key, value])
with open('SingleBandCutoff_'+str(n_cutoff)+'_M'+str(mz)+'N.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in data2.items():
       writer.writerow([key, value])



