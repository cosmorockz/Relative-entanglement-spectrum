from header import *
from Hamiltonians import ReferenceHamiltonianDiagonalizingMatrix
from common import *

def pair(list1,list2):
    # Contracts two lists with each other using fermionic anticommutation
    # Outputs all the possible contractions and the signs associated
    # E.g. list1 = [2,3] list2 = [4,6]
    # Output = [[(2, 4, 1), (3, 6, 1)], [(2, 6, -1), (3, 4, 1)]]
    # That means <2-4><3-6> - <2-6><3-4>
    if len(list1) == 1:
        return [[(list1[0],list2[0],1)]]
    element1 = list1[0]
    pairs = []
    for i in range(len(list2)):
        element2 = list2[i]
        subpairs = pair(list1[1:],list2[:i]+list2[i+1:])
        pairs = pairs + list(map(lambda x: [(element1,element2,(-1)**i)]+x,subpairs))
    return pairs

def BasisRotator(U,BasisElement):
    elements = BasisGeneratorNumberConserving(BasisElement)
    OpBasisElement = FockBasisToOperator(BasisElement)
    Column = np.zeros(2**(len(BasisElement)))
    # If the state is completely empty
    if OpBasisElement == []:
        BinaryString = ''.join(str(e) for e in BasisElement)
        Index = int(BinaryString,2)
        Column[Index] += 1
        return Column
    for element in elements:
        OpElement = FockBasisToOperator(list(element))
        contractions = pair(OpBasisElement,OpElement)
        MatrixElement = 0
        for possibility in contractions:
            # Loops through all the possible contractions
            term = 1
            for x in possibility:
                x = list(x)
                term = term * x[2] * U[x[0],x[1]]
            MatrixElement += term
        BinaryString = ''.join(str(e) for e in element)
        Index = int(BinaryString,2)
        Column[Index] += MatrixElement
    return Column

def FullRotation(M):
    mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
    URot = {}
    for m in mm: 
        basis = BasisGenerator(m)
        V = ReferenceHamiltonianDiagonalizingMatrix(m,M)
        U = np.zeros((len(basis),len(basis)))
        for v in basis:
            BinaryString = ''.join(str(e) for e in v)
            RowIndex = int(BinaryString,2)
            RotatorColumn = BasisRotator(V,v)
            for i, element in enumerate(RotatorColumn):
                U[RowIndex,i] = element
            URot[str(m)] = U
    return URot

def FullOrthogonalization():
    mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
    UOth = {}
    for m in mm: 
        basis = BasisGenerator(m)
        eigvals,V = OrthogonalizingMatrix(m)
        U = np.zeros((len(basis),len(basis)))
        for v in basis:
            BinaryString = ''.join(str(e) for e in v)
            RowIndex = int(BinaryString,2)
            RotatorColumn = BasisRotator(V,v)
            for i, element in enumerate(RotatorColumn):
                U[RowIndex,i] = element
            UOth[str(m)] = U
    return UOth





