#
# Le projet en lui-même
#
# Il présente tout
#

import numpy as np
import V1 as HMM


def sequence(adr):
    file = open(adr)
    S = []
    for word in file:
        w = []
        for char in word:
            w += [ord(char) - 97]
        S += [w]
    return S

def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -float('inf')
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        for i in range(1,nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2,n)]
            test = [S[l[j]] for j in range(f1,f2)]
            h = HMM.HMM.BW3(nbL,nbS,learn,nbIter,nbInit)
        lv += h.log_vraissemblance(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt,nbSOpt
