from header import *

def decimalToBinary(n):
    # Converts a given number in decimal representation to a binary number
    return "{0:b}".format(int(n))

def ContainerArray(n,ArraySize):
    # Given a decimal number, this function outputs a binary number
    # in an array of fixed legth.
    # For example, if n=4 and ArraySize=6,
    # The output will be [0,0,0,1,0,0]
    Container = [0 for i in range(ArraySize)]
    BinaryNumber = decimalToBinary(n)
    LenBinary = len(BinaryNumber)
    for i in range(LenBinary):
        Container[ArraySize-1-i] = int(BinaryNumber[LenBinary-1-i])
    return Container

def ManyBodyWavefunction(state):
    BinaryString = ''.join(str(e) for e in state)
    Index = int(BinaryString,2)
    TotalStates = len(state)
    ArraySize = 2**TotalStates
    S = [0 for i in range(ArraySize)]
    S[Index] = 1
    return np.array(S)

def BasisGenerator(m):
    # This function generates a full Dirac basis for a given angular
    # momentum m and a given cutoff
    s = int((2 * cutoff - 2*np.abs(m) + 1)/2)
    # Number of states with positive/negative energy. (total number of states)/2 for the given m and cutoff
    Tm = int(m + cutoff - 0.5) # Total number of angular momentums in front of the given angular momentum
    levels = 2 * s # Total number of levels for the given angular momentum
    MaxN = 2**levels # Highest number in the basis in decimal (By highest we mean the fully filled system)
    basis = []
    for i in range(MaxN):
        basis.append(ContainerArray(i,levels))
    return basis

def BasisGeneratorNumberConserving(BasisElement):
    # Given a basis element, this function generates all the other basis elements 
    # of same angular momentum and same particle number
    return list(set(list(itertools.permutations(BasisElement))))

def FockBasisToOperator(BasisElement):
    # Inputs a state in the Fock basis e.g. [0,1,0,1]
    # Outputs it as [2,4]
    basis = []
    for i,element in enumerate(BasisElement):
        if element == 1:
            basis.append(i)
    return basis

def OperatorToFockBasis(state):
    ArraySize = 2 * cutoff * (cutoff+1)
    Container = [0 for i in range(ArraySize)]
    for element in state:
        Container[element] = 1
    return Container

def DecimalToFockBasis(n):
    Binary = "{0:b}".format(int(n))
    ArraySize = 2 * cutoff * (cutoff+1)
    Container = [0 for i in range(ArraySize)]
    LenBinary = len(Binary)
    for i in range(LenBinary):
        Container[ArraySize-1-i] = int(Binary[LenBinary-1-i])
    return Container

def FockBasisToDecimal(state):
    BinaryString = ''.join(str(e) for e in state)
    n = int(BinaryString,2)
    return n

def OverlapMatrix(m):
    n = cutoff
    M = 2 * np.abs(m)
    N = 2 * n
    s = int((N-M+1)/2)
    OV = np.zeros((2*s,2*s))
    for i in range(s):
        for j in range(s):
            key1 = "-1,"+str(n-i)+",-1,"+str(n-j)+","+str(m)
            key2 = "-1,"+str(n-i)+",1,"+str(n-j)+","+str(m)
            key3 = "1,"+str(n-i)+",-1,"+str(n-j)+","+str(m)
            key4 = "1,"+str(n-i)+",1,"+str(n-j)+","+str(m)
            OV[i,j] = O[key4]
            OV[i,j+s] = O[key3]
            OV[i+s,j] = O[key2]
            OV[i+s,j+s] = O[key1]
    return OV

def OrthogonalizingMatrix(m):
    eigvals, eigvecs = np.linalg.eigh(OverlapMatrix(m))
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
    NewD = [U[i] for i in NewIndices]
    return np.array(NewE), np.array(NewD)

def OperatorTotalAM(state,MM):
    MM = np.array(MM)
    s = OperatorToFockBasis(state)
    TotalAm = np.dot(s,MM)
    return TotalAm



