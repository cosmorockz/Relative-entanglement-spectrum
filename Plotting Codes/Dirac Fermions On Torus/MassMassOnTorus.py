import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.linalg import null_space
import matplotlib.pyplot as plt

Ly = 12
nyCutoff = 6
Lx = 4
mz = 2
delta = 1e+9

def PlaneWave(y,ny):
    return np.exp(1j * (2 * np.pi * ny) * y / Ly) / np.sqrt(Ly)

def PlanewaveOverlaps(ny1,ny2,ThetaCut):
    RealPart = quad(lambda y: np.real(np.conjugate(PlaneWave(y,ny1)) * PlaneWave(y,ny2)), 0, ThetaCut)
    ImaginaryPart = quad(lambda y: np.imag(np.conjugate(PlaneWave(y,ny1)) * PlaneWave(y,ny2)), 0, ThetaCut)
    return RealPart[0] + 1j * ImaginaryPart[0]


def DiracWavefunction(nx,ny,la):
    
	kx = 2 * np.pi * nx / Lx
	ky = 2 * np.pi * ny / Ly
	kk = np.sqrt(kx**2 + ky**2)
	ek = np.sqrt(kk**2 + mz**2)
	if la == -1:
		if abs(kk) < 1e-10:
			comp1 = 0
		else:
			comp1 = np.sqrt(abs(ek-mz)) * (kx + 1j * ky)/kk
		comp2 = - np.sqrt(abs(ek+mz))
	if la == 1:
		comp1 = np.sqrt(abs(ek + mz))
		if abs(kk) < 1e-10:
			comp2 = 0
		else:
			comp2 = np.sqrt(abs(ek - mz)) * (kx - 1j * ky)/kk
    
	return np.array([comp1,comp2])/(np.sqrt(2 * ek))

def DiracOverlap(nx1,ny1,nx2,ny2,la1,la2,ThetaCut):
    PlanewavePart = PlanewaveOverlaps(ny1,ny2,ThetaCut)
    DiracWf1 = DiracWavefunction(nx1,ny1,la1)
    DiracWf2 = DiracWavefunction(nx2,ny2,la2)
    TotalOverlap = np.dot(np.conjugate(DiracWf1),DiracWf2) * PlanewavePart
    return TotalOverlap

def OverlapMatrix(nx,ThetaCut):
    s = 2 * nyCutoff + 1
    OV = np.zeros((2*s,2*s)) + 1j * np.zeros((2*s,2*s))
    kInd = - nyCutoff
    for i in range(s):
        lInd = - nyCutoff
        for j in range(s):
            temp1 = DiracOverlap(nx,kInd,nx,lInd,-1,-1,ThetaCut)
            if abs(temp1) > 1e-10:
                OV[i,j] = temp1
            temp2 = DiracOverlap(nx,kInd,nx,lInd,1,1,ThetaCut)
            if abs(temp2) > 1e-10:
                OV[i+s,j+s] = temp2
            temp3 = DiracOverlap(nx,kInd,nx,lInd,-1,1,ThetaCut)
            if abs(temp3) > 1e-10:
                OV[i,j+s] = temp3
            temp4 = DiracOverlap(nx,kInd,nx,lInd,1,-1,ThetaCut)
            if abs(temp4) > 1e-10:
                OV[i+s,j] = temp4
            lInd += 1
        kInd += 1
    return OV

def coefficients(nx,ThetaCut):
    OV = OverlapMatrix(nx,ThetaCut)
    eigvals1, eigvecs = np.linalg.eigh(OV)
    U = eigvecs
    U_T = np.linalg.inv(U)
    N = len(OV)
    N_A = U_T@OV@U
    OU = OV@U
    alpha = []
    for i in range(N):
        alpha.append([])
        for j in range(N):
            temp = OU[i, j]/np.sqrt(eigvals1[j])
            alpha[i].append(temp)
    O_B = np.eye(N) - OV
    eigvals2, eigvecs = np.linalg.eigh(O_B)
    U = eigvecs
    U_T = np.linalg.inv(U)
    N_B = U_T@O_B@U
    OU = O_B@U
    beta = []
    for i in range(N):
        beta.append([])
        for j in range(N):
            temp = OU[i, j]/np.sqrt(eigvals2[j])
            beta[i].append(temp)
    return alpha, beta

def EkRH(nx,ny):
    kx = 2 * np.pi * nx / Lx
    ky = 2 * np.pi * ny / Ly
    kk = np.sqrt(kx**2 + ky**2)
    ek = np.sqrt(kk**2 + mz**2)
    return ek

def DH(nx,ThetaCut):
    
    s = 2 * nyCutoff + 1
    energy = []
    for i in range(-nyCutoff,nyCutoff+1):
        energy.append(-EkRH(nx,i))
    for i in range(-nyCutoff,nyCutoff+1): 
        energy.append(EkRH(nx,i))
    
    CA, CB = coefficients(nx, ThetaCut)
    
    C = []
    for i in range(len(CA)):
        C.append([])
        for j in range(len(CA[0])):
            C[i].append(CA[i][j])
        for k in range(len(CB[0])):
            C[i].append(CB[i][k])
     
    H = np.zeros((2*len(C), 2*len(C))) + 1j * np.zeros((2*len(C), 2*len(C)))
    for i in range(len(C)):
        for j in range(len(H)):
            for k in range(len(H)):
                H[j, k] += energy[i] * np.conjugate(C[i][j]) * C[i][k]
     
    H = H + 1e-12 * np.eye(2*len(C))
     
    return H

def correlation_matrix(nx,ThetaCut):
    
    H = DH(nx,ThetaCut)
    eigvals, eigvecs = np.linalg.eigh(H)
    U = eigvecs.T
    U = np.linalg.inv(U)
    N = int(len(U)/2)
    
    C = np.zeros((N,N)) + 1j * np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(2*N):
                C[i,j] += np.conjugate(U[i,k]) * U[j,k] * np.heaviside(-float(eigvals[k]),0)
    
    return C

nxCutoff = int(Lx/2)
ThetaCut = Ly/2
NX = [-nxCutoff+i for i in range(2 * nxCutoff + 1)]

EE = []
NN = []

for nx in NX:
    C = DH(nx,ThetaCut) #correlation_matrix(nx,ThetaCut)
    eigvals, eigvecs = np.linalg.eigh(C)
    for j in range(len(eigvals)):
        # if eigvals[j] > 1e-20:
        EE.append(eigvals[j])
        NN.append(nx)

np.savetxt("Datamz"+str(mz)+"nyCutoff"+str(nyCutoff)+".dat",np.c_[NN,EE])

plt.scatter(NN,EE)
plt.grid()
plt.xlabel("$k_x$")
#plt.ylim(-0.2,1.2)
#plt.ylabel("$\lambda$")
plt.savefig("mz"+str(mz)+"nyCutoff"+str(nyCutoff)+".pdf")
plt.show()







