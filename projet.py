#######################
# Guillaume Augustoni #
#    Rania Alili      #
# Albane Durand-Viel  #
#   François Oder     #
#     projet.py       #
#  présentation des   #
# fonctionnalités de  #
#    la classe HMM    #
#######################


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
    nbSOpt = nbSMin
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        for i in range(1, nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2, n)]
            test = [S[l[j]] for j in range(f1, f2)]
            h = HMM.HMM.BW4_mieux(nbS, nbL, learn, nbInit)
            lv += h.log_vraisemblance(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return nbSOpt


def langue_prob(mot):
    w = sequence([mot])[0]
    HMMs = [HMM_neerland, HMM_allemand, HMM_espagnol, HMM_anglais]
    Langue = ['Néerlandais', 'Allemand', 'Espagnol', 'Anglais']
    logps = []
    for M in HMMs:
        logp = M.log_vraisemblance([w])
        logps += [logp]
    print(Langue[logps.index(max(logps))])


def mots_langue(M):
    S = []
    for i in range(10):
        w = M.gen_rand(3 + i % 5)[1]
        S += [w]
    print(mots(S))


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

logV_anglais1 = HMM_anglais1.log_vraisemblance(S_anglais)
print(logV_anglais1, "\n\n")

print("Cela nous permet de générer des mots potentiellement anglais :")
S = []
for i in range(10):
    w = HMM_anglais1.gen_rand(5)[1]
    S += [w]
print(mots(S))


print("\n\nPour déterminer le nombre d'états optimal pour notre HMM, nous avons exécuté le script suivant:")

Script = """
S_anglais = sequence_langue('anglais2000')
nbsOpt = xval(5, S_anglais, 26, 20, 50, 10)
print(nbsOpt)
"""


print(Script)
print("\n\nNous avons obtenu un nombre d'états optimal de 45, pour l'anglais. Étant donné le temps d'exécution de ce"
      "programme, nous avons supposé que c'était le même nombre d'états pour toutes les langues")


print("\n\nNous avons généré un HMM avec 45 états pour chaque langue, en tirant au hasard plusieurs HMMs à chaque fois,"
      " pour augmenter nos chances d'optenir un bon HMM\nNous avons  donc exécuté le script suivant, permettant de "
      "générer un HMM pour chaque langue et de le sauvegarder:\n\n")


Script = """
S_anglais = sequence_langue('anglais2000')
S_allemand = sequence_langue('allemand2000')
S_espagnol = sequence_langue('espagnol2000')
S_neerland = sequence_langue('neerland2000')


HMM_anglais = HMM.HMM.BW4_mieux(45, 26, S_anglais, 10)
print('anglais')
HMM_anglais.save('HMM_anglais')

HMM_allemand = HMM.HMM.BW4_mieux(45, 26, S_allemand, 10)
print('allemand')
HMM_allemand.save('HMM_allemand')

HMM_espagnol = HMM.HMM.BW4_mieux(45, 26, S_espagnol, 10)
print('espagnol')
HMM_espagnol.save('HMM_espagnol')


HMM_neerland = HMM.HMM.BW4_mieux(45, 26, S_neerland, 10)
print('neerland')
HMM_neerland.save('HMM_neerland')
"""


print(Script)
print("Maintenant, chargeons ces HMMs pour voir ce que nous pouvons en faire")

HMM_anglais = HMM.HMM.load('HMM_anglais')
HMM_allemand = HMM.HMM.load('HMM_allemand')
HMM_espagnol = HMM.HMM.load('HMM_espagnol')
HMM_neerland = HMM.HMM.load('HMM_neerland')


print("Une première utilité, que nous avons vue plus haut, est de générer pour chaque langue des mots pouvant y"
      " ressembler, de longueurs différentes par soucis de réalisme\n\nNous obtenons:")

print("\nPour l'anglais:\n")
mots_langue(HMM_anglais)

print("\nPour l'allemand:\n")
mots_langue(HMM_allemand)

print("\nPour l'espagnol:\n")
mots_langue(HMM_espagnol)

print("\nPour le néerlandais:\n")
mots_langue(HMM_neerland)


print("\n\nNous pouvons aussi déterminer la langue probable d'un mot (si nous avons un HMM pour la langue à laquelle il"
      "appartient). Voici quelques exemples:")

print("\n'five' est un mot anglais, on obtient: ", end='')
langue_prob('five')

print("\n'hablar' est un mot espagnol, on obtient: ", end='')
langue_prob('hablar')

print("\n'achtung' est un mot allemand, on obtient: ", end='')
langue_prob('achtung')

print("\n'miljoen' est un mot néerlandais, on obtient: ", end='')
langue_prob('miljoen')

print("\nCe modèle a des limites: 'los' est un mot espagnol, mais on obtient: ", end='')
langue_prob('los')
print("Car c'est aussi un mot anglais, et aussi néerlandais")
