import numpy as np
import random as rd


# Exo 11 question 1
class HMM:
    """ Define an HMM"""

    def __init__(self, nbl, nbs, initial, transitions, emissions):
        """

        :param nbl: nombre de lettres
        :param nbs: nombre d'états
        :param initial: les probabilités d'entrée dans le HMM
        :param transitions: les probabilités de transition à l'intérieur du HMM
        :param emissions: les probabilités d'émissions dans un HMM
        """
        self.nbl = nbl
        self.nbs = nbs
        self.initial = initial
        self.transitions = transitions
        self.emissions = emissions

    @property
    def nbl(self):
        """
        :return: Le nombre de lettres
        """
        return self.__nbl

    @nbl.setter
    def nbl(self, nbl):
        """
        La fonction modifie le nombre de lettres.
        :param nbl: Le nombre de lettres
        :return: None
        """
        if not isinstance(nbl, int):
            raise TypeError("nbl doit être entier")
        if nbl <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbl = nbl

    @property
    def nbs(self):
        """
        :return: Le nombre d'états
        """
        return self.__nbs

    @nbs.setter
    def nbs(self, nbs):
        """
        La fonction modifie le nombre d'états.
        :param nbs: Le nombre d'états
        :return: None
        """
        if not isinstance(nbs, int):
            raise TypeError("nbl doit être entier")
        if nbs <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbs = nbs

    @property
    def initial(self):
        """
        :return: Les probabilités d'entrée dans le HMM
        """
        return self.__initial

    @initial.setter
    def initial(self, initial):
        """
        :param initial: Les probabilités d'entrée dans le HMM
        :return: None
        """
        if not isinstance(initial, np.ndarray):
            raise TypeError("initial doit être un array numpy")
        if np.shape(initial) != (self.nbs, ):
            msg_err = "initial n'a pas la bonne dimension.\n"
            msg_err += "Sa dimension est " + str(np.shape(initial)) + ".\n"
            msg_err += "Dimension attendu :" + str((self.nbs,))
            raise ValueError(msg_err)
        if not np.isclose(np.array([initial.sum()]), np.array([1.0])):
            raise ValueError("la somme des probabilités initiales doit être 1")
        for pi in initial:
            if pi < 0:
                raise ValueError("les probabilités doivent être positives")
        self.__initial = initial

    @property
    def transitions(self):
        """
        :return: Les probabilités de transition à l'intérieur du HMM
        """
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions):
        """
        :param transitions: Les probabilités de transition à l'intérieur du HMM
        :return: None
        """
        if not isinstance(transitions, np.ndarray):
            raise TypeError("transitions doit être un array numpy")
        if np.shape(transitions) != (self.nbs, self.nbs):
            msg_err = "transitions n'a pas la bonne dimension.\n"
            msg_err += "Sa dimension est " + str(np.shape(transitions)) + ".\n"
            msg_err += "Dimension attendu :" + str((self.nbs, self.nbs))
            raise ValueError(msg_err)
        for line in range(self.nbs):
            if not np.isclose(np.array([transitions[line].sum()]), np.array([1.0])):
                raise ValueError("la somme des probabilités de transition, ligne par ligne, doit être 1")
        for line in transitions:
            for t in line:
                if t < 0:
                    raise ValueError("les probabilités doivent être positives")
        self.__transitions = transitions

    @property
    def emissions(self):
        """
        :return: Les proabilités d'émissions dans un HMM
        """
        return self.__emissions

    @emissions.setter
    def emissions(self, emissions):
        """
        :param emissions: Les probabilités d'émissions dans un HMM
        :return: None
        """
        if not isinstance(emissions, np.ndarray):
            raise TypeError("emissions doit être un array numpy")
        if np.shape(emissions) != (self.nbs, self.nbl):
            raise ValueError("emissions n'a pas la bonne dimension")
        for line in range(self.nbs):
            if not np.isclose(np.array([emissions[line].sum()]), np.array([1.0])):
                raise ValueError("la somme des probabilités de transition, ligne par ligne, doit être 1")
        for line in emissions:
            for e in line:
                if e < 0:
                    raise ValueError("les probabilités doivent être positives")
        self.__emissions = emissions

    @staticmethod
    def __lignes_moins_commentaire(adr):
        """
        :param adr: Adresse du fichier contenant la sauvegarde
        :return: Un itérateur contenant les lignes du fichier moins les lignes commençant par #
        """
        with open(adr, "r") as f:
            for line in f:
                if line[0] != "#":
                    yield line

# Exo 11 question 2
    @staticmethod
    def load(adr):
        """

        :param adr: l'adresse d'un fichier représentant un HMM
        :return: le HMM représenté dans le fichier
        """
        lignes = HMM.__lignes_moins_commentaire(adr)
        # Comme les fichiers ont toujours la même configuration moins les commentaires
        # On s'at
        nbl = int(next(lignes))
        nbs = int(next(lignes))
        initial = [float(next(lignes)) for _ in range(nbs)]
        transitions = [[j for j in map(float, next(lignes).split())] for _ in range(nbs)]
        emmissions = [[j for j in map(float, next(lignes).split())] for _ in range(nbl)]
        return HMM(nbl, nbs, np.array(initial).T, np.array(transitions), np.array(emmissions))

# Exo 11 question 4
    def save(self, adr='HMM.txt'):
        """

        :param adr: l'adresse où enregistrer le HMM self
        :return: None
        """
        fichier = open(adr, "w")
        fichier.write("# The number of letters \n")
        fichier.write(str(self.nbl))
        fichier.write("\n# The number of states \n")
        fichier.write(str(self.nbs))
        fichier.write("\n# The initial transitions \n")
        for i in self.initial:
            fichier.write(str(i) + "\n")
        fichier.write("# The internal transitions")
        for line in self.transitions:
            fichier.write("\n")
            for t in line:
                fichier.write(str(t) + ' ')
        fichier.write("\n# The emissions")
        for line in self.emissions:
            fichier.write("\n")
            for e in line:
                fichier.write(str(e) + ' ')
        fichier.close()

# Exo 11 question 3

    @staticmethod
    def draw_multinomial(l):
        """

        :param l: une liste de probabilités (sommant à 1)
        :return: un indice i avec une probabilité de l[i]
        """
        # x est une variable aléatoire comprise entre 0 et 1 suivant une distribution continue uniforme
        x = rd.random()
        somme_l = []
        somme = 0
        for i in range(len(l)):
            somme_l += [somme]
            somme += l[i]
        somme_l += [somme]
        for i in range(len(somme_l) - 1):
            if somme_l[i] <= x <= somme_l[i+1]:
                return i

    def gen_rand(self, n):
        """

        :param n: longueur du mot à générer
        :return: un mot généré à partir du HMM self
        """
        rd.seed()
        s = HMM.draw_multinomial(self.initial)
        S = [s]
        O = []
        for i in range(n):
            O += [HMM.draw_multinomial(self.emissions[s])]
            s = HMM.draw_multinomial(self.transitions[s])
            S += [s]
        return S, O

    def genere_f(self, w):
        """

        :param w: un mot généré par self
        :return: La matrice des f[k,i] qui sont les probabilités de générer w[:,i+1] en finissant sur l'état k
        """
        f = np.zeros((len(w), self.nbs))
        f[0] = self.initial*self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f[i] = (f[i - 1] @ self.transitions)*self.emissions[:, w[i]]

        return f.T

    def pfw(self, w):
        """
        :param w: séquence générée par le HMM self
        :return: la probabilité que self génère cette séquence
        """
        n = len(w)
        f = []
        for k in range(self.nbs):
            f += [self.initial[k]*self.emissions[k, w[0]]]
        f = np.array(f)
        for i in range(1, n):
            f = (f  @ self.transitions)*self.emissions[:, w[i]]
        return f.sum()

    def genere_b(self, w):
        """

        :param w: un mot généré par self
        :return: la matrice des b[k, i] qui sont les probabilités de générer w[i:] en partant de l'état k
        """
        assert len(w) != 0

        b = np.ones((self.nbs, len(w)))
        for i in reversed(range(len(w) - 1)):
            b[:,i] = self.transitions @ (self.emissions[:, w[i + 1]] * b[:, i + 1])

        return b

    def pbw(self, w):
        """
        :param w: séquence générée par le HMM self
        :return: la probabilité que self génère cette séquence
        """
        b = self.genere_b(w)
        return np.einsum("k,k,k->", self.initial, self.emissions[:, w[0]], b[:,0])
#Exo 14

    def viterbi(self, w):
        """

        :param w: une séquence
        :return: Le chemin de Viterbi de w et la probabilité associée
        """
        # on met les chemins et les probabilités dans un tableau de tuples

        chemins = []
        for k in range(self.nbs):
            chemin = [0]
            probabilité = 0
            for s in range(self.nbs):
                probabilité_t = self.emissions[(s, w[0])]*self.initial[s]*self.transitions[(s, k)]
                if probabilité_t >= probabilité:
                    chemin[0] = s
                    probabilité = probabilité_t
            chemins.append((chemin, probabilité))

        for i in range(1, len(w)):
            chemins_t = []
            for k in range(self.nbs):
                probabilité = 0

                for s in range(self.nbs):
                    probabilité_t = chemins[s][1]*self.emissions[(s, w[i])]
                    if i < (len(w) - 1):
                        probabilité_t *= self.transitions[(s, k)]
                    if probabilité_t >= probabilité:
                        chemin = (chemins[s][0] + [s], probabilité_t)
                        probabilité = probabilité_t
                chemins_t.append(chemin)
            chemins = chemins_t
        for i in range(len(chemins)):
            chem = (chemins[i][0], np.emath.log(chemins[i][1]))
            chemins[i] = chem
        return max(chemins, key=lambda x: x[1])

# Exo 15
    def pw_viterbi(self, w):
        """

        :param w: un mot généré par self
        :return: la probabilité de w le long de son chemin de viterbi
        """
        v = self.viterbi(w)
        res = 1
        for i in range(len(w)):
            res *= self.emissions[v[i], w[i]]
        return res

    def pviterbi_w(self, w):
        """

        :param w: Un mot généré par self
        :return: La probabilité du chemin de viterbi sachant que w a été généré
        """
        v = self.viterbi(w)
        p = self.pfw(w)
        pwv = self.initial[v[0]]
        for i in range(len(w)-1):
            pwv *= self.emissions[v[i], w[i]] * self.transitions[v[i], v[i+1]]
        pwv *= self.emissions[v[-1], w[-1]]
        return pwv/p

# Exo 16
    def nbviterbi(self, w, s):
        """

        :param w: un mot généré par self
        :param s: la liste des états cachés ayant généré w
        :return: le nombre d'états identiques dans s et dans le chemin de viterbi de w
        """
        v = self.viterbi(w)
        nb = 0
        for i in range(len(w)):
            if v[i] == s[i]:
                nb += 1
        return nb

# Exo 13
    def predit(self, w):
        """

        :param w: Un mot généré par self
        :return: La lettre qui a la plus forte probabilité d'être générée
        """
        H = self.initial
        for i in range(len(w)):
            H = (self.transitions.T*self.emissions[:, w[i]].T) @ H
        P = []
        for l in range(self.nbl):
            P += [self.emissions[:, l] @ H]
        return P.index(max(P))

    def log_vraissemblance(self, S):
        """

        :param S: une liste de mots potentiellements générés par self
        :return: le logarithme de la vraissemblance de self
        """
        res = 0
        for w in S:
            res += np.emath.log(self.pfw(w))
        return res

    @staticmethod
    def xi(m0, w, f, b):
        """

        :param m0: Le HMM que l'on veut modifier
        :param w: Un mot généré par self
        :param f: Une matrice de probabilités organisée de la manière suivante: f[k,i] est la probabilité d'arriver à la
        i-ème observation de w et que l'état final soit k
        :param b: La matrice des b[k, i] qui sont les probabilités de générer w[i:] en partant de l'état k
        :return: Une matrice de probabilités organisée de la manière suivante: xi[k,l,t] est la probabilité que le
        t-ième état soit k et que le t+1 ème état soit l, sachant que w est généré
        """
        f = f[:, :-1]
        b = b[:, 1:]
        emission = m0.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, m0.transitions, emission, b)
        norm = np.einsum('klt->t', xi)
        return xi/norm

    @staticmethod
    def gamma(f, b):
        return (f * b) / np.einsum('ki,ki->i', f, b)

    @staticmethod
    def BW1(m0, s):
        """

        :param m0:un HMM à mettre à jour pour augmenter la vraissemblance de s
        :type m0: HMM
        :param s: une liste de mots potentiellements générés par m0
        :return: la mise à jour de m0
        """
        pi = np.zeros(m0.nbs)
        T = np.zeros((m0.nbs, m0.nbs))
        O = np.zeros((m0.nbs, m0.nbl))
        for w in s:
            f = m0.genere_f(w)
            b = m0.genere_b(w)
            xi = HMM.xi(m0, w, f, b)
            gamma = HMM.gamma(f, b)
            pi += gamma[:, 0]
            T += np.einsum('klt->kl', xi)
            for o in range(m0.nbl):
                for t in range(len(w)):
                    if w[t] == o:
                        O[:, o] += gamma[:, t]

        # Normalisation
        pi /= pi.sum()
        for line in T:
            line /= line.sum()
        for line in O:
            line /= line.sum()

        return HMM(m0.nbl, m0.nbs, pi, T, O)

    @staticmethod
    def gen_HMM(nbs, nbl):
        """

        :param nbs: le nombre d'états du HMM à créer
        :param nbl: le nombre de lettres du HMM à créer
        :return: un HMM généré aléatoirement
        """
        rd.seed()
        somme = 0
        initial = []
        for i in range(nbs):
            x = rd.random()
            initial += [x]
            somme += x
        for i in range(nbs):
            initial[i] /= somme
        transitions = []
        for j in range(nbs):
            transitions += [[]]
            somme = 0
            for i in range(nbs):
                x = rd.random()
                transitions[j] += [x]
                somme += x
            for i in range(nbs):
                transitions[j][i] /= somme
        emissions = []
        for j in range(nbs):
            emissions += [[]]
            somme = 0
            for i in range(nbl):
                x = rd.random()
                emissions[j] += [x]
                somme += x
            for i in range(nbl):
                emissions[j][i] /= somme
        initial = np.array(initial)
        transitions = np.array(transitions)
        emissions = np.array(emissions)
        return HMM(nbl, nbs, initial, transitions, emissions)

    @staticmethod
    def BW2(nbs, nbl, S, N):
        """

        :param nbs: le nombre d'états du HMM à créer
        :param nbl: le nomre de lettres du HMM à créer
        :param S: une liste de mots pour laquelle on veut créer un HMM pour lequel la vraissemblance de S est grande
        :param N: le nombre de fois qu'on va mettre à jour un HMM tiré aléatoirement (avec BW1)
        :return: un HMM pour lequel la vraisemblance de S est très grande
        """
        M = HMM.gen_HMM(nbs, nbl)
        for i in range(N):
            M = HMM.BW1(M, S)
        return M

    @staticmethod
    def BW3(nbs, nbl, w, n, m):
        hmm_possibles = []
        for i in range(m):
            hmm_possibles += [HMM.BW2(nbs, nbl, [w], n)]
        return max(hmm_possibles, key=lambda x: x.pfw(w))

