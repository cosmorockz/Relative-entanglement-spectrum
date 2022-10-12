from header import *
from BlockSizeAM import BlockSize, LTotal, CBS

NLE = 40 # Number of low-energy eigenvalues
ESpectrum = []
MPoints = []
eps = 1e-6

EigsFolder = "./AMwiseEigs"

try:
    shutil.rmtree(EigsFolder)
except OSError:
    pass

os.makedirs(EigsFolder)

lock = threading.Lock()

def EigsCalculator(infoEigs):

    LL = infoEigs[0]
    blocksize = infoEigs[1]

    LDataFolder = DataFolder + "/L" + str(round(LL,1))
    print(LDataFolder)
    Nfiles = len(glob.glob(LDataFolder+"/*.csv"))
    print(Nfiles)
    RowsInfo, ColsInfo = [],[]
    DataInfo = []
    for j in range(Nfiles):
        DataFile = LDataFolder + "/L" + str(int(LL*1000)) + "file"+str(j)+".csv"
        IR, IC, ID = np.loadtxt(DataFile,usecols=(0,1,2),unpack=True,ndmin=1,dtype={'names':('rows','cols','data'),'formats':(np.int,np.int,np.float)})
        RowsInfo = RowsInfo + IR.tolist()
        ColsInfo = ColsInfo + IC.tolist()
        DataInfo = DataInfo + ID.tolist()
        
        
        del IR
        del IC
        del ID
        gc.collect()
    
    DataInfo = np.array(DataInfo)
    DataInfo[np.abs(DataInfo) < eps] = 0.0
        
    if RowsInfo != []:
        RHOM = sparse.csr_matrix((DataInfo,(RowsInfo,ColsInfo)),shape=(2**NumberofStates,blocksize))
        logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
        if blocksize <= NLE+1:
            eigvals = sp.linalg.svd(RHOM.todense(),compute_uv = False)
        else:
            eigvals = svds(RHOM,k=NLE,which="LM",return_singular_vectors=False,solver="arpack")
        
        del RHOM
        gc.collect()
        
        eigvals = eigvals.tolist()
        
        UsefulEigvals = [eigval for eigval in eigvals if np.abs(eigval) > eps]

        np.savetxt(EigsFolder+"/EsL"+str(int(LL*1000))+".csv",np.c_[-2 * np.log(np.array(UsefulEigvals))])
    
#         for eigval in eigvals:
#             MPoints.append(LL)
#             ESpectrum.append(eigval)
            
        del eigvals
        gc.collect()
    
    
    del RowsInfo
    del ColsInfo
    del DataInfo
    gc.collect()

    return None

for stat in zip(LTotal,BlockSize):
    EigsCalculator(stat)



