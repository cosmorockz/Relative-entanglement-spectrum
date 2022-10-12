from header import *

def energy(n,la):
    return la * n

def ReferenceHamiltonian(m,M):
    # Hamiltonian in the basis [+e1 +e2 +e3 +e4 .... .... -e1 -e2 -e3 -e4 .... ....]$
    # Where e1 > e2 > e3 > e4 > .... ....
    s = int((2*cutoff - 2*np.abs(m) + 1)/2)
    H = np.zeros((2 * s , 2 * s))
    for i in range(s):
        H[i,i] += cutoff - i
        H[i+s,i+s] += -(cutoff - i)
        H[i+s,i] += M
        H[i,i+s] += M 
    return H

def ReferenceHamiltonianDiagonalizingMatrix(m,M):
    # Returns the diagonalizing matrix in the basis [+e1 +e2 +e3 +e4 .... .... -e1 -e2 -e3 -e4 .... ....]
    # Where e1 > e2 > e3 > e4 > .... ....
    eigvals, eigvecs = np.linalg.eigh(ReferenceHamiltonian(m,M))
    U = eigvecs.T
    Indices = np.argsort(eigvals)
    NewIndices = []
    s = int(len(Indices)/2)
    for i in range(s):
        NewIndices.append(Indices[-i-1])
    for i in range(s):
        NewIndices.append(Indices[i])
#     U = np.linalg.inv(U)
    NewE = [eigvals[i] for i in NewIndices]
    NewD = []
    for i in NewIndices:
        maxU = np.argmax(np.abs(U[i]))
        sign = np.sign(U[i,maxU])
        for j in range(len(U[i])):
            U[i,j] = sign * U[i,j]
        NewD.append(U[i])
# #     U = np.array(NewD).T
# #     Uinv = np.linalg.inv(NewD)
# #     print(NewIndices)
# #     print(NewE)
    return np.array(NewD)


# def ReferenceHamiltonianDiagonalizingMatrix(m,M):
#     # Returns the diagonalizing matrix in the basis [+e1 +e2 +e3 +e4 .... .... -e1 -e2 -e3 -e4 .... ....]
#     # Where e1 > e2 > e3 > e4 > .... ....
#     eigvals, eigvecs = np.linalg.eigh(ReferenceHamiltonian(m,M))
#     U = eigvecs.T
#     Indices = np.argsort(eigvals)
#     NewIndices = []
#     s = int(len(Indices)/2)
#     for i in range(s):
#         NewIndices.append(Indices[-i-1])
#     for i in range(s):
#         NewIndices.append(Indices[i])
# #     U = np.linalg.inv(U)
#     NewE = [eigvals[i] for i in NewIndices]
#     NewD = [U[i] for i in NewIndices]
# # #     U = np.array(NewD).T
# # #     Uinv = np.linalg.inv(NewD)
# # #     print(NewIndices)
# # #     print(NewE)
#     return np.array(NewD)



