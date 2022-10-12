import csv
import numpy as np
from mpmath import *
from pathlib import Path

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
    for s in range(0, int(l-m+1)):
        if (l-Q >= s) and (l+Q >= l-m-s):
            sum_part += (-1)**s * binomial(l-Q, s) * binomial(l+Q, l-m-s) * \
                (v*v)**(l-Q-s) * (u*u)**s
    wf = pre * sum_part  # total wavefunction
    return wf

def FourComponentDiracWavefunction(theta,n,la,m,mz):
    e = np.sqrt(n**2 + mz**2)
    if la == -1:
        comp1 = 0
        comp2 = 0
        comp3 = sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
        comp4 = sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
    if la == 1:
        comp1 = sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
        comp2 = sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
        comp3 = 0
        comp4 = 0
    wf = 1/np.sqrt(2*e) * np.array([comp1,comp2,comp3,comp4])
    return wf

# def FourComponentDiracWavefunction(theta,n,la,m,mz):
#     e = np.sqrt(n**2 + mz**2)
#     if la == -1:
#         comp1 = sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
#         comp2 = - sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
#         comp3 = 0
#         comp4 = 0
#     if la == 1:
#         comp1 = sqrt(e+mz) * SLL(theta,phi,0.5,-0.5+n,m)
#         comp2 = sqrt(e-mz) * SLL(theta,phi,-0.5,-0.5+n,m)
#         comp3 = 0
#         comp4 = 0
#     wf = 1/np.sqrt(2*e) * np.array([comp1,comp2,comp3,comp4])
#     return wf

def OverlapIntegrand(theta,n1,n2,la1,la2,m,mz):
    wf1 = FourComponentDiracWavefunction(theta,n1,la1,m,mz)
    wf2 = FourComponentDiracWavefunction(theta,n2,la2,m,mz)
    OIntegrand = 2 * pi * sin(theta) * (np.conjugate(wf1[0])*wf2[0] + np.conjugate(wf1[1])*wf2[1] + np.conjugate(wf1[2])*wf2[2] + np.conjugate(wf1[3])*wf2[3])
    return OIntegrand

def overlap_integral(Q, n1, n2, la1, la2, m, theta_cut,mz):
    x = quad(lambda theta: OverlapIntegrand(
        theta, n1, n2, la1, la2, m,mz), [0, theta_cut])
    return x

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

def OverlapGenerator(n_cutoff,mz):
    ov_file = Path('DF_OI_Q_0_cutoff_'+str(n_cutoff)+"mz"+str(mz)+'.csv')
    if ov_file.is_file():
        print("Overlap Already Exists")
        reader = csv.reader(open(ov_file,'r'))
        OV = {}
        for k,v in reader:
            OV[k] = v
        return OV
    Q = 0
    Q_down = Q - 0.5
    Q_up = Q + 0.5

    theta_cut = pi/2
    mu = n_cutoff
    l_max = Q_down+n_cutoff
    mm = [-l_max+i for i in range(int(2*l_max+1))]

    data = {}

    for m in mm:
        levels, band = permitted_levels(n_cutoff,m,mu)
        N = len(levels)
        for i in range(N):
            for j in range(N):
                val = overlap_integral(Q,levels[i],levels[j],band[i],band[j],m,theta_cut,mz)
                if abs(val) < 1e-30:
                    val = 0
                k = str(band[i])+str(",")+str(levels[i])+str(",")+str(band[j])+str(",")+str(levels[j])+ str(",")+str(m)
                data[k] = val

    with open('DF_OI_Q_0_cutoff_'+str(n_cutoff)+"mz"+str(mz)+'.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in data.items():
           writer.writerow([key, value])
    
    reader = csv.reader(open('DF_OI_Q_0_cutoff_'+str(n_cutoff)+"mz"+str(mz)+'.csv','r'))
    OV = {}
    for k,v in reader:
        OV[k] = v
    return OV

OverlapGenerator(3,2)





