# francois oder le

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


S_anglais = sequence_langue('anglais2000')
S_allemand = sequence_langue('allemand2000')
S_espagnol = sequence_langue('espagnol2000')
S_neerland = sequence_langue('neerland2000')


HMM_anglais = HMM.HMM.load('HMM_anglais')
HMM_allemand = HMM.HMM.load('HMM_allemand')
HMM_espagnol = HMM.HMM.load('HMM_espagnol')
HMM_neerland = HMM.HMM.load('HMM_neerland')


HMM_anglais_W = HMM.HMM.load('HMM_anglais_moins_bon')
HMM_allemand_W = HMM.HMM.load('HMM_allemand_moins_bon')
HMM_espagnol_W = HMM.HMM.load('HMM_espagnol_moins_bon')
HMM_neerland_W = HMM.HMM.load('HMM_neerland_moins_bon')


logVanglais = HMM_anglais.log_vraisemblance(S_anglais)
logVanglais_W = HMM_anglais_W.log_vraisemblance(S_anglais)
logVallemand = HMM_allemand.log_vraisemblance(S_allemand)
logVallemand_W = HMM_allemand_W.log_vraisemblance(S_allemand)
logVespagnol = HMM_espagnol.log_vraisemblance(S_espagnol)
logVespagnol_W = HMM_espagnol_W.log_vraisemblance(S_espagnol)
logVneerland = HMM_neerland.log_vraisemblance(S_neerland)
logVneerland_W = HMM_neerland_W.log_vraisemblance(S_neerland)

print(logVanglais, logVanglais_W, logVallemand, logVallemand_W, logVespagnol, logVespagnol_W, logVneerland, logVneerland_W)
