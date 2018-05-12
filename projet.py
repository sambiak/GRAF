#
# Le projet en lui-même
#
# Il présente tout
#

import numpy as np
import HMM_classe as HMM


def sequence(adr):
    file = open(adr)
    S = []
    for word in file:
        w = []
        for char in word:
            w += [ord(char) - 97]
        w = w[:-1]
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


print("Bienvenue dans ce programme de présentation de la classe HMM", "\n")
print("Réalisée par le groupe GRAF", "\n\n")
print("Nous allons voir une application des HMM pour reconnnaître une langue et en générer des mots", "\n\n")
print("Commençons par récupérer une séquence de mots, les plus courants de la langue anglaise, que vous pouvez voir dans le fichier 'anglais2000'", "\n\n")
S_anglais = sequence('anglais2000')
print("Génèrons un HMM au hasard et adaptons-le à la génération des mots de cette séquence")
print("Nous avons choisis de générer un HMM avec 20 états, et de le mettre à jour 10 fois", "\n\n")
HMM_anglais1 = HMM.HMM.BW2(100, 26, S_anglais, 1000)
print("Nous obtenons un HMM que nous stockons dans le fichier 'HMM_anglais_V1'", "\n\n")
HMM_anglais1.save('HMM_anglais_V1')
print("Interressons-nous à la log-vraissemblance de notre séquence de mots anglais dans ce HMM:")
logV_anglais1 = HMM_anglais1.log_vraissemblance(S_anglais)
print(logV_anglais1)
