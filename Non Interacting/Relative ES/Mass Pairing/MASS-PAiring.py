import os
import csv
import numpy as np
import mpmath as mp
from mpmath import conj,eigh,matrix
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.linalg import block_diag

mp.dps = 30
n_cutoff = 10
mz = 4
Delta = - 4
delPair = 0
delta = 1e+10

reader1 = csv.reader(open('SingleBandCutoff_' +str(10) +'_M' +str(mz) +'N.csv', "r"))
reader2 = csv.reader(open('SingleBandCutoff_' +str(10) +'_M' +str(mz) +'P.csv', "r"))
O1 = {}
O2 = {}
for k,v in reader1:
    O1[k] = v
for k,v in reader2:
    O2[k] = v

if os.path.isdir('LowerHalf Mass'+str(mz)) == False:
    os.mkdir('LowerHalf Mass'+str(mz))

def OverlapMatrix(m):
    s = int((2*n_cutoff - 2*np.abs(m) + 1)/2)
    OV = mp.matrix(2*s,2*s)
    for i in range(s):
        for j in range(s):
            key = str(n_cutoff-s+1+i)+","+str(n_cutoff-s+1+j)+","+str(m)
            OV[i,j] = O1[key]
            OV[i+s,j+s] = O2[key]
    return OV

def coefficients(m):
    OV = OverlapMatrix(m)
    eigvals, eigvecs = mp.eigh(OV)
    U = eigvecs
    U_T = U.transpose()
    N = len(OV)
    N_A = U_T * OV * U
    OU = OV * U
    alpha = []
    for i in range(N):
        alpha.append([])
        for j in range(N):
            temp = OU[i, j]/mp.sqrt(eigvals[j])
            alpha[i].append(temp)
    O_B = mp.eye(N) - OV
    eigvals, eigvecs = mp.eigh(O_B)
    U = eigvecs
    U_T = U.transpose()
    N_B = U_T * O_B * U
    OU = O_B * U
    beta = []
    for i in range(N):
        beta.append([])
        for j in range(N):
            temp = OU[i, j]/mp.sqrt(eigvals[j])
            beta[i].append(temp)
    return alpha, beta

def EnP(n):
	e = mp.sqrt(n**2 + mz**2)
	en = (n**2 + Delta)/e
	return en
def EnN(n):
	e = mp.sqrt(n**2 + mz**2)
	en = (n**2 - Delta)/e
	return en
def Dn(n,m):
	e = mp.sqrt(n**2 + mz**2)
	dn = (n*Delta)/e * (-1)**(1/2 + m)
	return dn

def DH(m):
	
	s = int((2*n_cutoff - 2*np.abs(m) + 1)/2)
	energy1 = []
	for i in range(s):
		energy1.append(EnN(n_cutoff-s+1+i))
	energy2 = []
	for i in range(s):
		energy2.append(EnP(n_cutoff-s+1+i))
	pairing = []
	for i in range(s):
		pairing.append(Dn(n_cutoff-s+1+i,m))

	CA, CB = coefficients(m)
	C1 = []
	for j in range(len(CA)):
		C1.append([])
		for k in range(len(CA[0])):
			C1[j].append(CA[j][k])
			C1[j].append(CB[j][k])
	CA, CB = coefficients(-m)
	C2 = []
	for j in range(len(CA)):
		C2.append([])
		for k in range(len(CA[0])):
			C2[j].append(CA[j][k])
			C2[j].append(CB[j][k])

	# numpy coefficients for m
	A_m1 = np.zeros((len(CA),2*len(CA)))+1j*np.zeros((len(CA),2*len(CA)))
	for i in range(len(CA)):
		for j in range(2*len(CA)):
			A_m1[i,j] = C1[i][j]

    # orthogonal operators for m
	X_m = null_space(A_m1)
	X_m1 = X_m.T

    # numpy coefficients for -m
	A_m2 = np.zeros((len(CA),2*len(CA)))+1j*np.zeros((len(CA),2*len(CA)))
	for i in range(len(CA)):
		for j in range(2*len(CA)):
			A_m2[i,j] = C2[i][j]

    # orthogonal operators for -m
	X_m_n = null_space(A_m2)
	X_m2 = X_m_n.T

	H1 = mp.matrix(2*len(C1),2*len(C1))
	for i in range(s):
		for j in range(len(H1)):
			for k in range(len(H1)):
				H1[j,k] += energy1[i] * C1[i][k] * mp.conj(C1[i][j])
				H1[j,k] += energy2[i] * C1[i+s][k] * mp.conj(C1[i+s][j])
				H1[j,k] += delta * mp.conj(X_m1[i,k]) * X_m1[i,j]
				H1[j,k] += delta * mp.conj(X_m1[i+s,k]) * X_m1[i+s,j]

	H2 = mp.matrix(2*len(C2),2*len(C2))
	for i in range(s):
		for j in range(len(H2)):
			for k in range(len(H2)):
				H2[j,k] -= energy1[i] * mp.conj(C2[i][k]) * C2[i][j]
				H2[j,k] -= energy2[i] * mp.conj(C2[i+s][k]) * C2[i+s][j]
				H2[j,k] -= delta * mp.conj(X_m2[i,k]) * X_m2[i,j]
				H2[j,k] -= delta * mp.conj(X_m2[i+s,k]) * X_m2[i+s,j]

	D1 = mp.matrix(2*len(C1),2*len(C1))
	for i in range(s):
		for j in range(len(D1)):
			for k in range(len(D1)):
				D1[j,k] += (pairing[i] + delPair) * C1[i][j] * C2[i][k]
				D1[j,k] -= (pairing[i] - delPair) * C1[i+s][j] * C2[i+s][k]

	D2 = mp.matrix(len(D1),len(D1))
	for j in range(len(D2)):
		for k in range(len(D2)):
			D2[j,k] = -mp.conj(D1[k,j])

	HBDG = mp.matrix(2*len(H2),2*len(H2))
	for j in range(len(H2)):
		for k in range(len(H2)):
			HBDG[j,k] = H1[j,k]
			HBDG[j,k+len(H2)] = D1[j,k]
			HBDG[j+len(H2),k] = D2[j,k]
			HBDG[j+len(H2),k+len(H2)] = H2[j,k]

	return HBDG

def correlation_matrix(m):
    H = DH(m)
    eigvals, eigvecs = mp.eigh(H)
    U = eigvecs.transpose()
    U = U**-1
    N = int(len(U)/2)
    C = matrix(N,N)
    for i in range(N):
        for j in range(N):
            for k in range(2*N):
                C[i,j] += conj(U[2*i,k]) * U[2*j,k] * np.heaviside(-np.float(eigvals[k]),0)
    return C

mm = [-(n_cutoff-0.5)+i for i in range(2*n_cutoff)]

EE = []
M = []

for m in mm:
    C = correlation_matrix(m)
    eigvals, eigvecs = mp.eigh(C)
    for j in range(len(eigvals)):
        # if eigvals[j] > 1e-20:
        EE.append(eigvals[j])
        M.append(m)


np.savetxt("LowerHalf Mass"+str(mz)+"/Mass"+str(mz)+"Delta"+str(Delta)+"Cutoff"+str(n_cutoff)+".dat",np.c_[M,EE])

plt.figure(1)
plt.scatter(M,EE)
plt.grid()
plt.xlabel("m")
plt.ylabel("$\lambda$")
plt.title("$M_{Lower \; Half}=$"+str(mz)+",$Delta_{Upper \; Half}=$"+str(Delta))
# plt.savefig("M1_"+str(mz)+"M2_"+str(Mz)+"Cutoff"+str(n_cutoff)+".jpg")
plt.show()



