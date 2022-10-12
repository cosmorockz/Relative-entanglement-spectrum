import numpy as np
from mpmath import *
import matplotlib.pyplot as plt
import csv

mp.dps = 30
Q = 20
l_max = 20
mu = 0
reader = csv.reader(open('./overlaps/LL_OI_Q_'+str(Q)+'.csv','r'))
O = {}
for k,v in reader:
    O[k] = v

def overlap_matrix(l_max,m):
    # Returns the overlap matrix of the Landau levels after the entanglement cut
    n_ll = l_max-Q+1  # Number of filled Landau levels
    N = l_max-np.abs(m)+1  # Initiation for the overlap matrix
    if N >= n_ll:
        OV = matrix(n_ll, n_ll)
        for i in range(n_ll):
            for j in range(n_ll):
                key = str(Q+i)+','+str(Q+j)+','+str(m)
                OV[i, j] = O[key]

    elif N < n_ll:
        OV = matrix(N, N)
        for i in range(N):
            for j in range(N):
                key = str(np.abs(m)+i)+','+str(np.abs(m)+j)+','+str(m)
                OV[i, j] = O[key]
    return OV

def coefficients(l_max, m):
    '''
    Say c_1^{\dagger} and c_2^{\dagger} are the creation operators for
    the first two Landau levels. Now say a_1^{\dagger} and a_2^{\dagger}
    are the creation operators for orthonormal wavefunctions in the
    upper half sphere and b_1^{\dagger} and b_2^{\dagger} in the lower
    half sphere.
    c_1^{\dagger} = \alpha_{11} a_1^{\dagger} + \alpha_{12} a_2^{\dagger}
                    + \beta_{11} b_1^{\dagger} + \beta_{12} b_2^{\dagger}
    c_2^{\dagger} = \alpha_{21} a_1^{\dagger} + \alpha_{22} a_2^{\dagger}
                    + \beta_{21} b_1^{\dagger} + \beta_{22} b_2^{\dagger}

    So in general
    c_i^{\dagger} = \alpha_{ij} a_j^{\dagger} + \beta_{ij} b_j^{\dagger}

    The function returns alphas and betas as mpmath arrays. 
    '''
    OV = overlap_matrix(l_max, m)
    eigvals, eigvecs = eigh(OV)
    U = eigvecs
    U_T = U.T
    N = len(OV)
    N_A = U_T * OV * U
    OU = OV * U
    alpha = []
    for i in range(N):
        alpha.append([])
        for j in range(N):
            temp = OU[i, j]/sqrt(N_A[j, j])
            alpha[i].append(temp)
    O_B = eye(N) - OV
    eigvals, eigvecs = eigh(O_B)
    U = eigvecs
    U_T = U.T
    N_B = U_T * O_B * U
    OU = O_B * U
    beta = []
    for i in range(N):
        beta.append([])
        for j in range(N):
            temp = OU[i, j]/sqrt(N_B[j, j])
            beta[i].append(temp)
    return alpha, beta

def energy(l,Q):
    en = l*(l+1) - Q**2
    en = en/(2*np.abs(Q))
    return en

def Hamiltonian(CA,CB,l_max,m):
    en = []
    n_ll = l_max-Q+1 
    N = l_max-np.abs(m)+1
    if N >= n_ll:
        for i in range(n_ll):
            E = energy(Q+i,Q)
            en.append(E)
    elif N < n_ll:
        for i in range(N):
            E = energy(np.abs(m)+i,Q)
            en.append(E)
    
    C = []
    for i in range(len(CA)):
        C.append([])
        for j in range(len(CA[0])):
            C[i].append(sqrt(en[i])*CA[i][j])
        for k in range(len(CB[0])):
            C[i].append(sqrt(en[i])*CB[i][k])
    
    H = matrix(2*len(C), 2*len(C))
    for i in range(len(C)):
        for j in range(len(H)):
            for k in range(len(H)):
                H[j, k] -= conj(C[i][j]) * C[i][k]
    
    H = H + 1e-10 * eye(2*len(C))
    
    return H

def correlation_matrix(l_max, m):
    CA, CB = coefficients(l_max, m)
    H = Hamiltonian(CA, CB, l_max, m)
    eigvals, eigvecs = eigh(H)
    U = eigvecs.transpose()
    C = zeros(len(CA))
    for i in range(len(C)):
        for j in range(len(C)):
            for k in range(len(C)):
                # Uinv[i, k] * Uinvtranspose[j, k]
                C[i, j] += conj(U[i, k]) * U[j, k] * np.heaviside(mu-np.float(eigvals[k]),0)
    return C

l = l_max

E_prob = []

for m in range(-l, l+1):
    temp = correlation_matrix(l, m)
    eigvals, eigvecs = eigh(temp)
    E = eigvals
    E_prob.append(E)

E_prob_flat = []

for i in range(len(E_prob)):
    for j in range(len(E_prob[i])):
        E_prob_flat.append(E_prob[i][j])

M = []

for i in range(len(E_prob)):
    for j in range(len(E_prob[i])):
        M.append(i-l)

# Entanglement Spectrum

ES = np.array([log((1-E_prob_flat[i])/E_prob_flat[i])
               for i in range(len(E_prob_flat))])

np.savetxt("LL_ES_Q"+str(Q)+"_nu_"+str(l-Q+1)+".dat",np.c_[M,ES])

plt.figure(1)
plt.grid(which='both')
plt.scatter(M, ES, s=5)
plt.xlabel("m")
plt.ylabel("Pseudoenergy")
plt.minorticks_on()
plt.title("Q="+str(Q)+",$\\nu=$"+str(l-Q+1))
# plt.title("Single Particle Entanglement Spectrum for filling factor " +
#           str(l-Q+1)+",Q="+str(Q)+",entanglement cut at pi/2")
# plt.savefig("LL_ES_Q"+str(Q)+"_nu_"+str(l-Q+1)+".pdf")
plt.show()
