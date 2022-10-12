import gc
from header import *
from common import *
from Rotation import FullRotation, FullOrthogonalization
from PH import phTransform
import matplotlib.pyplot as plt
from BlockSizeAM import BlockSize, LTotal


def PSWfSegmentor(wf):
    # !Product State Wavefuntion segmentor
    # Segments a product state wafunction into 
    # Angular momentum sectors
    wf = np.array(wf)
    wf = wf.tolist()
    WF = {}
    mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
    for m in mm:
        s = int((2*cutoff - 2*np.abs(m) + 1))
        temp = []
        for i in range(s):
            temp.append(wf.pop(0))
        WF[str(m)] = ManyBodyWavefunction(temp)
    return WF

# Probing wavefunction in the Fock basis (massive Mz basis)
# wf = [0,1,0,1]
wf = [0,1,0,0,1,1,0,0,1,1,0,1]
# wf = [0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,1]
WF = PSWfSegmentor(wf) # Written in occupation number basis, separated angular momentum-wise

# Rotation matrices
URot1 = FullRotation(Mz) # Rotation matrix for mass Mz
URot2 = FullRotation(mz) # Rotation matrix for mass mz

# Rotate the wavefunction to the massless basis
WFMR1 = {}
for k,v in WF.items():
    U = URot1[k]
    U = np.linalg.inv(U)
    WFMR1[str(k)] = U@WF[k]

# Rotate the wavefunction to the massive basis (massive mz basis)
WFMR2 = {}
for k,v in WFMR1.items():
    U = URot2[k]
    WFMR2[str(k)] = U@WFMR1[k]

# Wavefunction in the sparse format
WFMR_sparse = {}
for k,v in WFMR2.items():
    WFMR_sparse[k] = sparse.csr_matrix(v)

fullWf = []
for k,v in WFMR_sparse.items():
    if fullWf == []:
        fullWf = v
    else:
        fullWf = sparse.kron(fullWf,v)

fullWf = fullWf.tocoo() # Full wavefunction in the coo format
# Coo  column index, row indexm, value for non-zero index
rows = fullWf.row
cols = fullWf.col
data = fullWf.data

del wf
del WF
del WFMR1
del WFMR2
del WFMR_sparse
del fullWf
gc.collect()

# Non-zero entries in the sparse format
rowsNZ = [] # Non-zero row index
colsNZ = [] # Non-zero column index
dataNZ = [] # Non-zeroo data

for i,d in enumerate(data):
    if abs(d) > tol:
        rowsNZ.append(rows[i])
        colsNZ.append(cols[i])
        dataNZ.append(d)

phCols = []
phData = []
for i,col in enumerate(colsNZ):
    state = DecimalToFockBasis(col)
    sign1, state = phTransform(state)
    colInd = FockBasisToDecimal(state)
    phCols.append(colInd)
    phData.append(sign1*dataNZ[i])

rowsNZ = np.array(rowsNZ)
phCols = np.array(phCols)
phData = np.array(phData)

'''Orthogonalizing the basis in the half sphere'''
UOth = FullOrthogonalization()

# Full half-sphere orthogonalised wavefunction in the sparse format
OrthWf = []
for i,col in enumerate(phCols):
    stateFock = DecimalToFockBasis(col) # Nonzero column entry to Fock basis
    segWF = PSWfSegmentor(stateFock) # Angular momentum resolved wavefunction
    wfOrth = {} # Orthogonalized wavefunction, corresponding to the column
    wfOrFull = [] # Full half-sphere orthogonalised wavefunction, corresponding to the column
    for k,v in segWF.items():
        U = UOth[k]
        wfOrth[k] = sparse.csr_matrix(U@v)
        if wfOrFull == []:
            wfOrFull = wfOrth[k]
        else:
            wfOrFull = sparse.kron(wfOrFull,wfOrth[k])
            
    if OrthWf == []:
        OrthWf = phData[i] * wfOrFull
    else:
        OrthWf = OrthWf + phData[i] * wfOrFull
    
#     del wfOrFull

'''Non-zero entries of the particle-hole transformed half-sphere
orthogonalized wavefunction'''

OrthWf = OrthWf.tocoo()
rows = OrthWf.row
cols = OrthWf.col
data = OrthWf.data

del OrthWf
gc.collect()


# Non-zero entries of the orthogonalized wavefunction in the sparse format

rowsNZ = []
colsNZ = []
dataNZ = []
for i,d in enumerate(data):
    if abs(d) > tol:
        rowsNZ.append(rows[i])
        colsNZ.append(cols[i])
        dataNZ.append(data[i])

# del rows
# del cols
# del data
# gc.collect()


'''Decomposition'''

def OperatorDecomposition(state):
    # Decomposes an array into two arrays in all possible ways
    # Given a wavefunction in the operator basis, it decomposes the wavefunction in A and B parts in all the possible ways
    flags = [False]*len(state)
    D = []
    while True:
        a = [state[i] for i,flag in enumerate(flags) if flag]
        b = [state[i] for i,flag in enumerate(flags) if not flag]
        D.append([a,b])
        for i in range(len(state)):
            flags[i] = not flags[i]
            if flags[i]:
                break
        else:
            break
    return D


overlapUA = [] # Overlap (half-sphere) eigenvalues in A part of the sphere
overlapUB = [] # Overlap (half-sphere) eigenvalues in B part of the sphere
mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
MM = [] # Array of angular momentum in every site of the Fock basis
for m in mm:
    eigvals, eigvecs = OrthogonalizingMatrix(m)
    for i in range(len(eigvals)):
        overlapUA.append(np.sqrt(eigvals[i]))
        overlapUB.append(np.sqrt(1-eigvals[i]))
    s = int((2*cutoff - 2*np.abs(m) + 1))
    for i in range(s):
        MM.append(m)


# Constructing the blocks of density matrices whose columns are A elements and rows are B elements

RhoBlocks = {}
FullMatrix = np.zeros((4096,4096))
for i, col in enumerate(colsNZ):
    
    IndFock = DecimalToFockBasis(col)
    IndOper = FockBasisToOperator(IndFock)
    DD = OperatorDecomposition(IndOper)
    
    for a,b in DD:
        
        aFock = OperatorToFockBasis(a)
        bFock = OperatorToFockBasis(b)
        aInd = FockBasisToDecimal(aFock)
        bInd = FockBasisToDecimal(bFock)
        MatrixElement = dataNZ[i]
        
        for element in a:
            MatrixElement = overlapUA[element] * MatrixElement
            
        for element in b:
            MatrixElement = overlapUB[element] * MatrixElement
        
        FullMatrix[bInd,aInd] += MatrixElement
        
# #         aTotalAm = OperatorTotalAM(a,MM)
# #         kM = str(aTotalAm) # Key for the highest level of the dictionary
        
# #         if str(kM) not in RhoBlocks.keys():
# #             RhoBlocks[kM] = {}
# #             RhoBlocks[kM]['rows'] = [bInd]
# #             RhoBlocks[kM]['cols'] = [aInd]
# #             RhoBlocks[kM]['data'] = [MatrixElement]
# #         else:
# #             RhoBlocks[kM]['rows'].append(bInd)
# #             RhoBlocks[kM]['cols'].append(aInd)
# #             RhoBlocks[kM]['data'].append(MatrixElement)

# Array of Angular momentum in every site of the basis
mm = [-(cutoff-0.5)+i for i in range(int(2*(cutoff-0.5)+1))]
M = []
for m in mm:
    s = int((2*cutoff - 2*np.abs(m) + 1)/2)
    for i in range(s):
        M.append(m)
        M.append(m)
LargestNumber = 2**(2*cutoff*(cutoff+1))
BasisAm = []
for i in range(LargestNumber):
    State = DecimalToFockBasis(i)
    State = np.array(State)
    AM = np.dot(State,M)
    BasisAm.append(AM)
Ind = np.argsort(BasisAm)
FullMatrix = FullMatrix.T
FMordered = np.array([FullMatrix[i] for i in Ind])
FMordered = FMordered.T
RHOA = np.matmul(FMordered.T,FMordered)

CBS = [BlockSize[0]]
for i in range(len(BlockSize)-1):
    CBS.append(CBS[i]+BlockSize[i+1])
BD = [RHOA[0:CBS[0],0:CBS[0]]]
for i in range(len(CBS)-1):
    TM = RHOA[CBS[i]:CBS[i+1],CBS[i]:CBS[i+1]]
    BD.append(TM)
L = LTotal
LL = []
ES = []
EE = []
for i,R in enumerate(BD):
#     if BlockSize[i] > 1:
#         eigvals, eigvecs = eigsh(R,k=7) #np.linalg.eigh(R)
#     else:
#         eigvals, eigvecs = np.linalg.eigh(R)
    eigvals, eigvecs = np.linalg.eigh(R)
    for eigval in eigvals:
#         if eigval > 1e-10:
        es = - np.log(eigval)
        LL.append(L[i])
        ES.append(es)
        EE.append(eigval)

plt.scatter(np.array(LL),ES)
plt.grid()
plt.show()

# LowE = [] # Low energy eigenvalues
# CorM = [] # Corresponding total angular momenta

# EE = []
# tr = 0

# for key, value in RhoBlocks.items():
    
#     RowM = value['rows']
#     ColM = value['cols']
#     DatM = value['data']
    
#     setRow = set(RowM)
#     sortedRow = sorted(setRow)
#     rankedRow = []
#     for i,e in enumerate(RowM):
#         rankedRow.append(sortedRow.index(e))
    
#     setCol = set(ColM)
#     sortedCol = sorted(setCol)
#     rankedCol = []
#     for i,e in enumerate(ColM):
#         rankedCol.append(sortedCol.index(e))
    
#     BlockM = sparse.coo_matrix((DatM,(rankedRow,rankedCol)))
#     print(BlockM.shape)
#     BlockM = BlockM.tocsr()
#     BlockMdagger = BlockM.getH()
#     RhoAAM = BlockMdagger.multiply(BlockM)
    
# #     if RhoAAM.shape[0] == 1:
# #         RhoAAM = RhoAAM.todense()
# #         eigvals, eigvecs = np.linalg.eigh(RhoAAM)
# #     else:
# #         eigvals, eigvecs = eigsh(RhoAAM,k=(RhoAAM.shape[0]-1),which='LM')
    
#     # For testing only
#     RHOAAM = RhoAAM.todense()
#     tr += np.trace(RHOAAM)
#     eigvals,eigvecs = np.linalg.eigh(RHOAAM)
    
#     for i,eigval in enumerate(eigvals):
#         EE.append(eigval)
# #         LowE.append(-np.log(eigval))
#         CorM.append(float(key))

# Etot = sum(EE)
# EEnormalised = np.array(EE)/Etot
# ES = -np.log(EEnormalised)
    

# # del RhoAAM
# # del RhoBlocks
        


    
    
    



# FW = fullWf.toarray()
# print(FW)
# print(np.sum(FW**2))
# print(len(FW[0]))
# print(len(np.nonzero(FW[0])[0]))






