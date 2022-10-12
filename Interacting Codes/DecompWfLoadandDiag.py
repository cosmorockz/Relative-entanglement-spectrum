import gc
from header import *
from BlockSizeAM import BlockSize, LTotal, CBS, DecimalToSortedDecimal
import matplotlib.pyplot as plt

NLE = 200 # Number of low-energy eigenvalues
ESpectrum = []
MPoints = []
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
    Nfiles = len(os.listdir(LDataFolder))
    RowsInfo, ColsInfo = [],[]
    DataInfo = []
    for j in range(Nfiles):
        DataFile = LDataFolder + "/L" + str(int(LL*1000)) + "file"+str(j)+".csv"
        IR, IC, ID = np.loadtxt(DataFile,usecols=(0,1,2),unpack=True,ndmin=1,dtype={'names':('rows','cols','data'),'formats':(np.int,np.int,np.float)})
        print("L="+str(LL)+"File="+str(j)+"Memory="+str(process.memory_info().rss/(1024**3)))
        logging.debug("L="+str(LL)+"File="+str(j)+"Memory="+str(process.memory_info().rss/(1024**3))+"\n")
        RowsInfo = RowsInfo + IR.tolist()
        ColsInfo = ColsInfo + IC.tolist()
        DataInfo = DataInfo + ID.tolist()

        del IR
        del IC
        del ID
        gc.collect()


    if RowsInfo != []:
        RHOM = sparse.csr_matrix((DataInfo,(RowsInfo,ColsInfo)),shape=(2**NumberofStates,blocksize))
        RHOA = RHOM.T @ RHOM
        print("Matrix Size="+str(RHOA.get_shape()))
        logging.debug("Matrix Size="+str(RHOA.get_shape()))
        logging.debug("Angular Momentum = "+str(LL)+"Memory Consumption = "+str(process.memory_info().rss/(1024**3))+" \n")
        if blocksize <= NLE+1:
            eigvals, eigvecs = np.linalg.eigh(RHOA.todense())
            print(process.memory_info().rss/(1024**3))
            logging.debug(process.memory_info().rss/(1024**3))
        else:
            eigvals, eigvecs = eigsh(RHOA,k=NLE,sigma=1,which="SA")
            print(process.memory_info().rss/(1024**3))
            logging.debug(process.memory_info().rss/(1024**3))

        np.savetxt(EigsFolder+"/EsL"+str(int(LL*1000))+".csv",np.c_[-np.log(eigvals)])

        del RHOM
        del RHOA
        gc.collect()

        # for eigval in eigvals:
        #     MPoints.append(LL)
        #     ESpectrum.append(-np.log(eigval))

        # del eigvals
        # del eigvecs
        # gc.collect()

    return None

for infoeigs in zip(LTotal,BlockSize):
	EigsCalculator(infoeigs)







