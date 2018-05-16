#
# Le projet en lui-même
#
# Il présente tout
#

import numpy as np
import HMM_classe as HMM


def sequence_langue(adr):
    file = open(adr)
    res = sequence(file)
    file.close()
    return res

def sequence(S):
    Res = []
    for word in S:
        w = []
        for char in word:
            w += [ord(char) - 97]
        w = w[:-1]
        Res += [w]
    return Res

def mots(S):
    Res = []
    for w in S:
        word = ''
        for ch in w:
            char = chr(ch + 97)
            word += char
        Res += [word]
    return Res

def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbInit):
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
            h = HMM.HMM.BW4_mieux(nbS,nbL,learn,nbInit)
        lv += h.log_vraissemblance(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt,nbSOpt

"""
print("Bienvenue dans ce programme de présentation de la classe HMM", "\n")
print("Réalisée par le groupe GRAF", "\n\n")
print("Nous allons voir une application des HMM pour reconnnaître une langue et en générer des mots", "\n\n")
print("Commençons par récupérer une séquence de mots, les plus courants de la langue anglaise, que vous pouvez voir dan"
      "s le fichier 'anglais2000'", "\n\n")
S_anglais = sequence_langue('anglais2000')
print("Générons un HMM au hasard et adaptons-le à la génération des mots de cette séquence.")
print("Nous avons choisi de générer un HMM avec 30 états.", "\n\n")
HMM_anglais1, it = HMM.HMM.BW2_mieux(30, 26, S_anglais)
print("Nous obtenons un HMM que nous stockons dans le fichier 'HMM_anglais_V1'.", "\n\n")
HMM_anglais1.save('HMM_anglais_V1')
print("Nous avons mis à jour notre HMM " + str(it) + " fois avant d'obtenir des HMMs avec des vraisemblances stables")
print("Intéressons-nous à la log-vraisemblance de notre séquence de mots anglais dans ce HMM:")
logV_anglais1 = HMM_anglais1.log_vraissemblance(S_anglais)
print(logV_anglais1, "\n\n")
print("Cela nous permet de générer des mots potentiellement anglais :")
S = []
for i in range(10):
    w = HMM_anglais1.gen_rand(5)[1]
    S += [w]
print(mots(S))
"""

S_anglais = sequence_langue('anglais2000')
S_allemand = sequence_langue('allemand2000')
S_espagnol = sequence_langue('espagnol2000')
S_neerland = sequence_langue('neerland2000')


"""
nbs_anglais = xval(5, S_anglais, 26, 20, 100, 10)
print(nbs_anglais)
nbs_allemand = xval(5, S_allemand, 26, 20, 100, 10)
print(nbs_allemand)
nbs_espagnol = xval(5, S_espagnol, 26, 20, 100, 10)
print(nbs_espagnol)
nbs_neerland = xval(5, S_neerland, 26, 20, 100, 10)
print(nbs_neerland)
"""


HMM_anglais = HMM.HMM.BW4_mieux(30, 26, S_anglais, 10)
print('anglais')
HMM_anglais.save('HMM_anglais')

HMM_allemand = HMM.HMM.BW4_mieux(30, 26, S_allemand, 10)
print('allemand')
HMM_allemand.save('HMM_allemand')

HMM_espagnol = HMM.HMM.BW4_mieux(30, 26, S_espagnol, 10)
print('espagnol')
HMM_espagnol.save('HMM_espagnol')

HMM_neerland = HMM.HMM.BW4_mieux(30, 26, S_neerland, 10)
print('neerland')
HMM_neerland.save('HMM_neerland')

