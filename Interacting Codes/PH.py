from header import *
from common import *
from sympy.combinatorics.permutations import Permutation

mm = [-(cutoff-0.5)+i for i in range(2*cutoff)]
WfIndDict = {} # Fock basis wavefunction index dictionary
indi = 0
for m in mm:
    s = int((2*cutoff - 2*np.abs(m) + 1)/2)
    for i in range(s):
        k = str(indi)
        indi += 1
        WfIndDict[k] = {}
        WfIndDict[k]['n'] = cutoff - i
        WfIndDict[k]['lambda'] = '+'
        WfIndDict[k]['m'] = m
    for i in range(s):
        k = str(indi)
        indi += 1
        WfIndDict[k] = {}
        WfIndDict[k]['n'] = cutoff - i
        WfIndDict[k]['lambda'] = '-'
        WfIndDict[k]['m'] = m

WfIndReverseDict = {} # Fock basis wavefunction reverse dictionary
PositiveStates = [] # All the positive operators index
NegativeStates = [] # All the negative operators index
for k,v in WfIndDict.items():
    n = v['n']
    l = v['lambda']
    m = v['m']
    NewKey = 'n'+str(n)+'l'+l+'m'+str(m)
    WfIndReverseDict[NewKey] = k
    if l == '-':
        NegativeStates.append(int(k))
    else:
        PositiveStates.append(int(k))
PositiveStates = sorted(PositiveStates)
NegativeStates = sorted(NegativeStates)

def ConjugateAmLocationN(loca):
    # Given a Fock basis location, finds the location of -m if the energy is negative
    # Otherwise returns the same location
    l = WfIndDict[str(loca)]['lambda']
    if l == '+':
        return loca
    n = WfIndDict[str(loca)]['n']
    m = WfIndDict[str(loca)]['m']
    k = 'n'+str(n)+'l'+l+'m'+str(-m)
    return int(WfIndReverseDict[k])

lenN = len(NegativeStates) # Total number of negative operators for a given cutoff
phMatrix = np.zeros((lenN,lenN))
for i,e in enumerate(NegativeStates):
    location = ConjugateAmLocationN(e)
    indi = NegativeStates.index(location)
    phMatrix[i,indi] = 1

def ph(n):
    try:
        nIndex = NegativeStates.index(n)
    except:
        return None
    nState = phMatrix@np.array(NegativeStates)
    return int(nState[nIndex])

def phTransform(wf):
    wf = FockBasisToOperator(wf)
    wfPos = list(set(wf) & set(PositiveStates)) # Filled Positive states
    wfNeg = list(set(wf) & set(NegativeStates)) # Filled Negative states
    phNeg = sorted(list(set(NegativeStates) - set(wfNeg))) # Negative energy states that are not filled
    
    wfFullNP = wfNeg + wfPos
    sIndices = (np.argsort(wfFullNP))
    oIndices = list(np.argsort(sIndices))
    S1 = Permutation(oIndices).signature()
    
    wfFullNN = [e-0.5 for e in wfNeg] + NegativeStates
    sIndices = (np.argsort(wfFullNN))
    oIndices = list(np.argsort(sIndices))
    S2 = Permutation(oIndices).signature()
    
    Holes = [ph(i) for i in phNeg]
    wfFullHP = Holes + wfPos
    sIndices = (np.argsort(wfFullHP))
    oIndices = list(np.argsort(sIndices))
    S3 = Permutation(oIndices).signature()
    
    tWfOp = sorted(wfPos+Holes) # Total particle hole transformed wavefunction in the operator basis
    tWf = OperatorToFockBasis(tWfOp) # Total particle hole transformed wavefunction in the Fock basis
    
    return S1*S2*S3, tWf





