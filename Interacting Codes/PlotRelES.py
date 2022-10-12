import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from BlockSizeAM import LTotal

EigsFolder = "./AMwiseEigs"

MPoints = []
ES = []

for LL in LTotal:
    FileName = Path(EigsFolder + "/EsL" + str(int(LL*1000)) + ".csv")
    if FileName.is_file():
        es = np.loadtxt(FileName,unpack=True,ndmin=1)
        for e in es:
            MPoints.append(LL)
            ES.append(e)
    
np.savetxt("RelativeES.dat",np.c_[MPoints,ES])

plt.scatter(MPoints,ES,s=8)
# plt.xlim(-5,5)
# plt.ylim(10,20)
plt.grid()
# plt.savefig("M2M-2InteractingCutoff3FullLowe.pdf")
plt.show()



