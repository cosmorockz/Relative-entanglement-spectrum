import numpy as np
from header import *
import pandas as pd
from pathlib import Path

NumberofStates = 2 * cutoff * (cutoff + 1)

def decimalToBinary(n):
    return "{0:b}".format(int(n))
LargestNumber = 2**(NumberofStates)-1
ArraySize = len(decimalToBinary(LargestNumber))

def ContainerArray(n):
    Container = np.array([0 for i in range(ArraySize)])
    BinaryNumber = decimalToBinary(n)
    LenBinary = len(BinaryNumber)
    for i in range(LenBinary):
        Container[ArraySize-1-i] = BinaryNumber[LenBinary-1-i]
    return Container

mm = [-(cutoff-0.5)+i for i in range(int(2*(cutoff-0.5)+1))]
M = []
for m in mm:
    s = int((2*cutoff - 2*np.abs(m) + 1)/2)
    for i in range(s):
        M.append(m)
        M.append(m)

data = {}
AMlist = []

def ComputeAm(n):
    
    Number = ContainerArray(n)
    AngularMomentum = np.dot(M,Number)
    AMlist.append(AngularMomentum)
    ParticleNumber = np.sum(Number)
    key1 = str(ParticleNumber)
    key2 = str(AngularMomentum)
    if key1 in data:
        if key2 in data[key1]:
            data[key1][key2] = data[key1][key2] + 1
        else:
            data[key1][key2] = 1
    else:
        data[key1] = {}
        data[key1][key2] = 1
   
    return None


BlockSizeFile = Path("./BlockSizeFileCutoff"+str(cutoff)+".dat")

CBSFile = Path("./CBSCutoff"+str(cutoff)+".dat")

LTotalFile = Path("./LTotalCutoff"+str(cutoff)+".dat")

AmListFile = Path("./AmListCutoff"+str(cutoff)+".dat")

if BlockSizeFile.is_file() and CBSFile.is_file() and LTotalFile.is_file() and AmListFile.is_file():
    print("Blocks already Computed before")
    BlockSize = np.loadtxt(BlockSizeFile,unpack=True)
    BlockSize = BlockSize.astype(int)
    CBS = np.loadtxt(CBSFile,unpack=True)
    CBS = CBS.astype(int)
    LTotal = np.loadtxt(LTotalFile,unpack=True)
    AMlist = np.loadtxt(AmListFile,unpack=True)
else:

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = executor.map(ComputeAm,range(LargestNumber+1))
        
    for i in range(0,LargestNumber+1):
        ComputeAm(i)

    Data = pd.DataFrame(data)
    Data = Data.fillna(0)
    Data = Data.transpose()
    x = list(Data.columns)
    x = [float(x[i]) for i in range(len(x))]
    x.sort()
    x = [str(x[i]) for i in range(len(x))]
    Data = Data.reindex(x, axis=1)

    DataNumpy = Data.to_numpy()
    DataNumpy = DataNumpy.T

    BlockSize = np.array([int(sum(DataNumpy[i])) for i in range(len(DataNumpy))])
    LTotal = np.array(list(set(AMlist)))
    LTotal = np.sort(LTotal)

    CBS = [BlockSize[0]]
    for i in range(len(BlockSize)-1):
        CBS.append(CBS[i]+BlockSize[i+1])
    CBS = [0] + CBS

    CBS.pop()

    np.savetxt("BlockSizeFileCutoff"+str(cutoff)+".dat",np.c_[BlockSize])

    np.savetxt("CBSCutoff"+str(cutoff)+".dat",np.c_[CBS])

    np.savetxt("LTotalCutoff"+str(cutoff)+".dat",np.c_[LTotal])

    np.savetxt("AmListCutoff"+str(cutoff)+".dat",np.c_[AMlist])
    
    del data
    del Data
    del x
    del DataNumpy
    gc.collect()

IndAm1 = np.argsort(AMlist)
IndAm2 = np.argsort(IndAm1)

del AMlist
del IndAm1

gc.collect()

def DecimalToSortedDecimal(state):
    return IndAm2[state]




