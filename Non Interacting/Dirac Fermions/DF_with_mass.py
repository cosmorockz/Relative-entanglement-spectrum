import numpy as np
from mpmath import *
import matplotlib.pyplot as plt
import csv


mp.dps = 30
n_cutoff = 20
en_const = 1
m_z = 2
reader = csv.reader(open('DF_OI_Q_0_cutoff_30.csv','r'))
O = {}
for k,v in reader:
    O[k] = v

def energy_init(la,n):
    temp = en_const * la * n
    return temp

def H_nm(n,m):
    h11 = energy_init(-1,n)
    h12 = 2 * m_z
    h21 = 2 * m_z
    h22 = energy_init(1,n)
    H = matrix([[h11,h12],[h21,h22]])
    eigvals, eigvecs = eigh(H)
    U = eigvecs.transpose()
    return eigvals, U

def overlap_matrix(n_cutoff,m):
    n = n_cutoff
    M = 2 * np.abs(m)
    N = 2 * n
    s = int((N-M+1)/2)
    OV = matrix(2*s,2*s)
    for i in range(s):
        for j in range(s):
            eigvals1, U1 = H_nm(n-i,m)
            eigvals2, U2 = H_nm(n-j,m)
            key1 = "-1,"+str(n-i)+",-1,"+str(n-j)+","+str(m)
            key2 = "1,"+str(n-i)+",-1,"+str(n-j)+","+str(m)
            key3 = "-1,"+str(n-i)+",1,"+str(n-j)+","+str(m)
            key4 = "1,"+str(n-i)+",1,"+str(n-j)+","+str(m)
            o1 = float(O[key1])
            o2 = float(O[key2])
            o3 = float(O[key3])
            o4 = float(O[key4])
            OV[i,j] = conj(U1[0,0]) * U2[0,0] * o1 + conj(U1[0,1]) * U2[0,0] * o2 + conj(U1[0,0]) * U2[0,1] * o3 + conj(U1[0,1]) * U2[0,1] * o4
            OV[i+s,j] = conj(U1[1,0]) * U2[0,0] * o1 + conj(U1[1,1]) * U2[0,0] * o2 + conj(U1[1,0]) * U2[0,1] * o3 + conj(U1[1,1]) * U2[0,1] * o4
            OV[i,j+s] = conj(U1[0,0]) * U2[1,0] * o1 + conj(U1[0,1]) * U2[1,0] * o2 + conj(U1[0,0]) * U2[1,1] * o3 + conj(U1[0,1]) * U2[1,1] * o4
            OV[i+s,j+s] = conj(U1[1,0]) * U2[1,0] * o1 + conj(U1[1,1]) * U2[1,0] * o2 + conj(U1[1,0]) * U2[1,1] * o3 + conj(U1[1,1]) * U2[1,1] * o4
    return OV

def coefficients(n_cutoff, m):
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
    OV = overlap_matrix(n_cutoff, m)
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

def energy(la,n):
	if la == -1:
		en = -sqrt((en_const*n)**2 + m_z**2)
	if la == 1:
		en = sqrt((en_const*n)**2 + m_z**2)
	return en

def Hamiltonian(CA,CB,n_cutoff,m):
    
    M = 2 * np.abs(m)
    N = 2 * n_cutoff
    s = int((N-M+1)/2)
    en = []
    for i in range(s):
        en.append(energy(-1,n_cutoff-i))
    for i in range(s):
        en.append(energy(1,n_cutoff-i))
    
    C = []
    for i in range(len(CA)):
        C.append([])
        for j in range(len(CA[0])):
            C[i].append(CA[i][j])
        for k in range(len(CB[0])):
            C[i].append(CB[i][k])
    
    H = matrix(2*len(C), 2*len(C))
    for i in range(len(C)):
        for j in range(len(H)):
            for k in range(len(H)):
                H[j, k] += en[i] * conj(C[i][j]) * C[i][k]
    
    H = H + 1e-20 * eye(2*len(C))
    
    return H

def correlation_matrix(n_cutoff, m):
    CA, CB = coefficients(n_cutoff, m)
    H = Hamiltonian(CA, CB, n_cutoff, m)
    eigvals, eigvecs = eigh(H)
    U = eigvecs.transpose()
    U = U**-1
    C = zeros(int(len(CA)))
    for i in range(len(C)):
        for j in range(len(C)):
            for k in range(2*len(C)):
                C[i, j] += conj(U[i, k]) * U[j, k] * np.heaviside(-np.float(eigvals[k]),0)
    return C

Q = 0
Q_down = Q - 0.5
Q_up = Q + 0.5
mu = 0
l_max = Q_down+n_cutoff
mm = [-l_max+i for i in range(int(2*l_max+1))]

E_prob = []

for m in mm:
    temp = correlation_matrix(n_cutoff,m)
    eigvals, eigvecs = eigh(temp)
    E = eigvals
    E_prob.append(E)


E_prob_flat = []

for i in range(len(E_prob)):
    for j in range(len(E_prob[i])):
        if abs(E_prob[i][j]) > 1e-20:
            E_prob_flat.append(E_prob[i][j])

for i in range(len(E_prob_flat)):
    if E_prob_flat[i] < -1:
        E_prob_flat[i] = -1
    if E_prob_flat[i] > 1:
        E_prob_flat[i] = 1

M = []

for i in range(len(mm)):
    for j in range(int(n_cutoff-abs(mm[i])+1/2)):
        M.append(mm[i])
#         M.append(mm[i])

ES = np.array([(log((1-E_prob_flat[i])/E_prob_flat[i])) for i in range(len(E_prob_flat))])
for i in range(len(ES)):
    if abs(ES[i]) < 1e-15:
        ES[i] = 0


plt.figure(1)
plt.scatter(M, ES)
plt.grid()
plt.xlabel("m")
plt.ylabel("Pseudoenergy")
# plt.savefig("plots_and_dats/ES_DF_m_"+str(m_z)+"_cutoff_"+str(n_cutoff)+".pdf")
plt.show()

ES = np.array(ES.tolist(),dtype=float)
np.savetxt("ES_DF_m_"+str(m_z)+"_cutoff_"+str(n_cutoff)+".dat",np.c_[M,ES])



