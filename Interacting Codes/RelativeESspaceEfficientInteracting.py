from header import *
from common import *
import csv
from Rotation import FullRotation, FullOrthogonalization
from PH import phTransform
from BlockSizeAM import BlockSize, LTotal, CBS, DecimalToSortedDecimal
from sympy.combinatorics.permutations import Permutation
import concurrent.futures
from joblib import Parallel, delayed

AMCounterReset = {}

for LL in LTotal:
    
    AMCounterReset[round(LL,1)] = 0

lock = threading.Lock()

try:
    os.remove("loggingData.log")
except OSError:
    pass

try:
    shutil.rmtree(DataFolder)
except:
    pass

os.makedirs(DataFolder)

for LL in LTotal:
    
    LFolder = DataFolder+"/L"+str(round(LL,1))
    try:
        shutil.rmtree(LFolder)
    except OSError:
        pass
    
    os.makedirs(LFolder)

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

def phTransform2(n):
    return phTransform(DecimalToFockBasis(n))

basis = np.array(list(csv.reader(open("./Cutoff3/basisU30c3.csv", "r"), delimiter=","))).astype("float")
state = np.array(list(csv.reader(open("./Cutoff3/stateEvenU30c3.csv", "r"), delimiter=","))).astype("float")

mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
NumberofStates = 2 * cutoff * (cutoff + 1)

BasisDictr = {}
k = 0
for m in mm:
    s = int((2*cutoff - 2*np.abs(m) + 1)/2)
    for i in range(s):
        key1 = float(cutoff-i)
        key = str(m)+ "," +str(key1)
        BasisDictr[key] = k
        k = k +1
    for i in range(s):
        key1 = float(-(cutoff-i))
        key = str(m)+ "," +str(key1)
        BasisDictr[key] = k
        k = k +1

InitState = [0 for i in range(2**NumberofStates)]
InitRows = []
InitCols = []
InitData = []
xx = []
for i in range(len(state)):
    Nparticles = int(NumberofStates/2)
    x = []
    for j in range(Nparticles):
        key = str(basis[j,2*i+1])+ "," +str(basis[j,2*i])
        x.append(BasisDictr[key])
    sIndices = (np.argsort(x))
    oIndices = list(np.argsort(sIndices))
    signPE = Permutation(oIndices).signature()
    Fock = OperatorToFockBasis(x)
    Deci = FockBasisToDecimal(Fock)
#     if abs(state[i][0]) > toll:
    InitRows.append(0)
    InitCols.append(Deci)
    InitData.append(signPE*state[i][0])

InitData = InitData/np.sum(np.array(InitData)**2)
del state
del basis
gc.collect()

URot = FullRotation(mz)

def StateRotation(state):
    state = DecimalToFockBasis(state)
    WfSp = PSWfSegmentor(state)
    fullWf = []
    for k,v in WfSp.items():
        if fullWf == []:
            fullWf = URot[k]@WfSp[k]
        else:
            fullWf = sparse.kron(fullWf,URot[k]@WfSp[k])
#     WfRo = WfRo + coef * fullWf
    return fullWf

WfRo = sparse.csr_matrix(([0],([0],[0])),shape=(1,2**NumberofStates))
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(StateRotation,InitCols)
    
    for i,result in enumerate(results):
        WfRo += InitData[i] * result

WfRo = WfRo.tocoo()
WfRoRows = WfRo.row
WfRoCols = WfRo.col
WfRoData = WfRo.data
del WfRo
del URot
gc.collect()

WfPhRows = WfRoRows
WfPhCols = []
WfPhData = []

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(phTransform2,WfRoCols.tolist())
    
    for i,result in enumerate(results):
        WfPhCols.append(FockBasisToDecimal(result[1]))
        WfPhData.append(result[0]*WfRoData[i])

del WfRoCols
del WfRoData
gc.collect()

print("Rotation Done")

UOth = FullOrthogonalization()

def StateOrthogonalizer(state):
    state = DecimalToFockBasis(state)
    WfSp = PSWfSegmentor(state)
    fullWf = []
    for k,v in WfSp.items():
        if fullWf == []:
            fullWf = UOth[k]@WfSp[k]
        else:
            fullWf = sparse.kron(fullWf,UOth[k]@WfSp[k])
    return fullWf

WfOt = sparse.csr_matrix(([0],([0],[0])),shape=(1,2**NumberofStates))

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(StateOrthogonalizer,WfPhCols)
    
    for i, result in enumerate(results):
        WfOt += WfPhData[i] * result

logging.debug("Orthogonalization done \n")

print(process.memory_info().rss/(1024**3))
print("Orthogonalization done")
        
del WfPhCols
del WfPhData
gc.collect()

WfOt = WfOt.tocoo()
WfOtCols = WfOt.col
WfOtData = WfOt.data
del WfOt
del UOth
gc.collect()

logging.debug("Non-zero elements in the wavefunction " + str(len(WfOtCols)) + "\n")

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

def ElementwiseDecomposition(state):
    
    col = state[0]
    data = state[1]
#     aIndex = state[2] # Index of the total angular momentum
    
    colRows = []
    colCols = []
    colData = []
    
    IndFock = DecimalToFockBasis(col)
    IndOper = FockBasisToOperator(IndFock)
    DD = OperatorDecomposition(IndOper)
    for a,b in DD:
        aFock = OperatorToFockBasis(a)
        bFock = OperatorToFockBasis(b)
        aInd = FockBasisToDecimal(aFock)
        bInd = FockBasisToDecimal(bFock)
        aAM = np.dot(aFock,MM) # Total columner angular momentum
        c = a + b
        sIndices = (np.argsort(c))
        oIndices = list(np.argsort(sIndices))
        signPE = Permutation(oIndices).signature()
        MatrixElement = data
        for element in a:
            MatrixElement *= overlapUA[element]            
        for element in b:
            MatrixElement *= overlapUB[element]
        
        aIndex = LTotal.tolist().index(round(aAM,1))
        colsInfo = DecimalToSortedDecimal(aInd) - CBS[aIndex]
        
        if abs(MatrixElement) > MatrixElementtol:
            with lock:
                AMCounterReset[round(aAM,1)] += 1
                filename = DataFolder + "/L" + str(round(aAM,1)) + "/L" + str(int(aAM*1000)) + "file"+str(int((AMCounterReset[round(aAM,1)]-1)/maxFileSize))+".csv" 
                with open(filename,"a") as fd:
                    fd.write("%d %d %.4E \n" % (bInd, colsInfo, signPE * MatrixElement))
    
    if gcProb >= np.random.uniform(0,1):
        del a
        del b
        del aFock
        del bFock
        del aInd
        del bInd
        del aAM
        del c
        del sIndices
        del oIndices
        del signPE
        del MatrixElement
        del aIndex
        del colsInfo
        del col
        del data
        del colRows
        del colCols
        del colData
        del IndFock
        del IndOper
        del DD
        gc.collect()
        print("Deallocation Done")

    return None

logging.debug("Pre decompoistion steps done \n")

print(process.memory_info().rss/(1024**3))
print("Pre decompoistion steps done")

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(ElementwiseDecomposition,zip(WfOtCols,WfOtData))
#         for result in results:
#             pass

        
        
        
        
        

# for i,state in enumerate(zip(WfOtCols,WfOtData)):
#     ElementwiseDecomposition(state)

# SectorwiseRelESComputation()

# for i,LL in enumerate(LTotal):
# #     if LL > -2 and LL < 2:
#     if abs(LL) < 1e-10:
#         if BlockSize[i] <= NLE + 1:
#             IR, IC, ID = SectorwiseRelEsComputation(i)
#             RHOM = sparse.csr_matrix((ID,(IR,IC)),shape=(2**NumberofStates,BlockSize[i]))
#             RHOA = RHOM.T@RHOM # Block Matrix
#             logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
#             print(process.memory_info().rss/(1024**3))
#             eigvals, eigvecs = np.linalg.eigh(RHOA.todense())
#         else:
#             IR, IC, ID = SectorwiseRelEsComputation(i)
#             RHOM = sparse.csr_matrix((ID,(IR,IC)),shape=(2**NumberofStates,BlockSize[i]))
#             RHOA = RHOM.T@RHOM # Block Matrix
#             logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
#             print(process.memory_info().rss/(1024**3))
#             eigvals, eigvecs = eigsh(RHOA,k=NLE,sigma=1,which="SA")
#         for eigval in eigvals:
#             MPoints.append(LL)
#             ESpectrum.append(-np.log(eigval))

# print("Decomposition and diagonalization done")

# np.savetxt("ES_mz"+str(mz)+"_Mz"+str(Mz)+"cutoff"+str(cutoff)+".dat",np.c_[MPoints,ESpectrum])

# plt.scatter(MPoints,ESpectrum)
# plt.grid()
# # plt.ylim(-1,10)
# plt.xlabel("$L_z^{tot}$")
# plt.ylabel("Pseudoenergy")       
# plt.title("mz="+str(mz)+",Cutoff="+str(cutoff))
# plt.savefig("ES_mz"+str(mz)+"_Mz"+str(Mz)+"cutoff"+str(cutoff)+".pdf")

# plt.show()




# NLE = 40
# ESpectrum = []
# MPoints = []
# for i, LL in enumerate(LTotal):
#     LDataFolder = DataFolder + "/L" + str(round(LL,1))
#     if BlockSize[i] <= NLE+1:
#         DataFile = LDataFolder + "/L" + str(int(LL*1000)) + "file0.csv"
#         IR, IC, ID = np.loadtxt(DataFile,usecols=(0,1,2),ndmin=1,unpack=True,dtype={'names':('rows','cols','data'),'formats':(np.int,np.int,np.float)})
#         RHOM = sparse.csr_matrix((ID,(IR,IC)),shape=(2**NumberofStates,BlockSize[i]))
#         RHOA = RHOM.T @ RHOM
#         logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
# #         print(process.memory_info().rss/(1024**3))
#         eigvals, eigvecs = np.linalg.eigh(RHOA.todense())
#     else:
#         DataFile = LDataFolder + "/L" + str(int(LL*1000)) + "file0.csv"
#         IR, IC, ID = np.loadtxt(DataFile,usecols=(0,1,2),unpack=True)
#         RHOM = sparse.csr_matrix((ID,(IR,IC)),shape=(2**NumberofStates,BlockSize[i]))
#         RHOA = RHOM.T @ RHOM
#         logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
# #         print(process.memory_info().rss/(1024**3))
#         eigvals, eigvecs = eigsh(RHOA,k=NLE,sigma=1,which="SA")
#     for eigval in eigvals:
#         MPoints.append(LL)
#         ESpectrum.append(-np.log(eigval))

# NLE = 40
# ESpectrum = []
# MPoints = []
# for i, LL in enumerate(LTotal):
#     print(LL, BlockSize[i])
#     LDataFolder = DataFolder + "/L" + str(round(LL,1))
#     Nfiles = len(os.listdir(LDataFolder))
#     RowsInfo, ColsInfo = [],[]
#     DataInfo = []
#     for j in range(Nfiles):
#         DataFile = LDataFolder + "/L" + str(int(LL*1000)) + "file"+str(j)+".csv"
#         IR, IC, ID = np.loadtxt(DataFile,usecols=(0,1,2),unpack=True,ndmin=1,dtype={'names':('rows','cols','data'),'formats':(np.int,np.int,np.float)})
#         RowsInfo = RowsInfo + IR.tolist()
#         ColsInfo = ColsInfo + IC.tolist()
#         DataInfo = DataInfo + ID.tolist()
#     RHOM = sparse.csr_matrix((DataInfo,(RowsInfo,ColsInfo)),shape=(2**NumberofStates,BlockSize[i]))
#     RHOA = RHOM.T @ RHOM
#     logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
#     if BlockSize[i] <= NLE+1:
#         eigvals, eigvecs = np.linalg.eigh(RHOA.todense())
#     else:
#         eigvals, eigvecs = eigsh(RHOA,k=NLE,sigma=1,which="SA")
#     for eigval in eigvals:
#         MPoints.append(LL)
#         ESpectrum.append(-np.log(eigval))



