import numpy as np
from mpmath import *
import csv

mp.dps = 30
Delta1 = 6
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

def FourComponentDiracWavefunction(theta,n,la,m):
    e = np.sqrt(n**2 + Delta1**2)
    if la == -1:
        comp1 = 0
        comp2 = - np.conjugate(Delta1)/e * (-1)**(m+1/2) * SLL(theta,0,-1/2,n-1/2,m)
        comp3 = (-1)**(m+1/2) * SLL(theta,0,-1/2,n-1/2,m)
        comp4 = - (-1)**(m-1/2) * n/e * SLL(theta,0,1/2,n-1/2,m)
    if la == 1:
        comp1 = SLL(theta,0,1/2,n-1/2,m)
        comp2 = n/e * SLL(theta,0,-1/2,n-1/2,m)
        comp3 = 0
        comp4 = Delta1/e * SLL(theta,0,1/2,n-1/2,m)
    wf = 1/np.sqrt(2) * np.array([comp1,comp2,comp3,comp4])
    return wf

def OverlapIntegrand(theta,n1,n2,la1,la2,m):
    wf1 = FourComponentDiracWavefunction(theta,n1,la1,m)
    wf2 = FourComponentDiracWavefunction(theta,n2,la2,m)
    OIntegrand = 2 * pi * sin(theta) * (np.conjugate(wf1[0])*wf2[0] + np.conjugate(wf1[1])*wf2[1] + np.conjugate(wf1[2])*wf2[2] + np.conjugate(wf1[3])*wf2[3])
    return OIntegrand

def OverlapIntegral(n1,n2,la1,la2,m,ThetaCut):
    OIntegral = quad(lambda theta: OverlapIntegrand(theta,n1,n2,la1,la2,m),[0,ThetaCut])
    return OIntegral

def permitted_levels(n_cutoff,m,mu):
    # This function outputs the allowed levels for a given m, chemical
    # potential and cutoff

    mu = int(mu)
    M = int(abs(m)+0.5)
    levels = []
    band = []

    if mu == 0:
        for i in range(n_cutoff,M-1,-1):
            band.append(-1)
            levels.append(i)
    
    if mu > 0:
        for i in range(n_cutoff,M-1,-1):
            band.append(-1)
            levels.append(i)
        for i in range(M,mu+1):
            band.append(1)
            levels.append(i)
    
    if mu < 0:
        for i in range(n_cutoff,max(abs(mu)-1,M-1),-1):
            band.append(-1)
            levels.append(i)

    return levels, band

n_cutoff = 10
Q_down = - 0.5
theta_cut = pi/2
mu = 10
l_max = Q_down+n_cutoff
mm = [-l_max+i for i in range(int(2*l_max+1))]

data = {}

for m in mm:
    levels, band = permitted_levels(n_cutoff,m,mu)
    N = len(levels)
    for i in range(N):
        for j in range(N):
            val = OverlapIntegral(levels[i],levels[j],band[i],band[j],m,theta_cut)
            if abs(val) < 1e-30:
                val = 0
            k = str(band[i])+str(",")+str(levels[i])+str(",")+str(band[j])+str(",")+str(levels[j])+ str(",")+str(m)
            data[k] = val

with open('PairingOverlaps'+str(n_cutoff)+'Delta'+str(Delta1)+'.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in data.items():
       writer.writerow([key, value])



