import numpy as np
from header import *
import pandas as pd

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
for n in range(0,LargestNumber+1):
    Number = ContainerArray(n)
    AngularMomentum = 0
    for i in range(len(M)):
        AngularMomentum += M[i] * Number[i]
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



