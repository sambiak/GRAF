import numpy as np
import random as rd


# Exo 11 question 1
class HMM:
    """ Define an HMM"""

    def __init__(self, nbl, nbs, initial, transitions, emissions):
        # The number of letters
        self.nbl = nbl
        # The number of states
        self.nbs = nbs
        # The vector defining the initial weights
        self.initial = initial
        # The array defining the transitions
        self.transitions = transitions
        # The list of vectors defining the emissions
        self.emissions = emissions

    @property
    def nbl(self):
        return self.__nbl

    @nbl.setter
    def nbl(self, nbl):
        if not isinstance(nbl, int):
            raise ValueError("nbl doit être entier")
        if nbl <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbl = nbl

    @property
    def nbs(self):
        return self.__nbs

    @nbs.setter
    def nbs(self, nbs):
        if not isinstance(nbs, int):
            raise ValueError("nbl doit être entier")
        if nbs <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbs = nbs

    @property
    def initial(self):
        return self.__initial

    @initial.setter
    def initial(self, initial):
        if not isinstance(initial, np.ndarray):
            raise ValueError("initial doit être un array numpy")
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
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions):
        if not isinstance(transitions, np.ndarray):
            raise ValueError("transitions doit être un array numpy")
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
        return self.__emissions

    @emissions.setter
    def emissions(self, emissions):
        if not isinstance(emissions, np.ndarray):
            raise ValueError("emissions doit être un array numpy")
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
    def __ligns_not_comments(adr):
        """
        :param adr: Adresse du fichier contenant la sauvegarde
        :return: Un itérateur contenant les lignes du fichier moins les lignes
        commencant par #
        """
        with open(adr, "r") as f:
            for line in f:
                if line[0] != "#":
                    yield line

# Exo 11 question 2
    @staticmethod
    def load(adr):
        lines = HMM.__ligns_not_comments(adr)
        nbl = int(next(lines))
        nbs = int(next(lines))
        initial = [float(next(lines)) for _ in range(nbs)]
        transitions = [[j for j in map(float, next(lines).split())] for _ in range(nbs)]
        emmissions = [[j for j in map(float, next(lines).split())] for _ in range(nbl)]
        return HMM(nbl, nbs, np.array(initial).T, np.array(transitions), np.array(emmissions))

# Exo 11 question 4
    def save(self, adr='HMM.txt'):
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
    def draw_multinomial(L):
        x = rd.random()
        M = []
        somme = 0
        for i in range(len(L)):
            M += [somme]
            somme += L[i]
        M += [somme]
        for i in range(len(M) - 1):
            if M[i] <= x <= M[i+1]:
                return i

    def gen_rand(self, n):
        rd.seed()
        s = HMM.draw_multinomial(self.initial)
        S = [s]
        O = []
        for i in range(n):
            O += [HMM.draw_multinomial(self.emissions[s])]
            s = HMM.draw_multinomial(self.transitions[s])
            S += [s]
        return O

    def pfw(self, w):
        """
        :param w: séquence générée par le HMM self
        :return: la probabilité que self génère cette séquence
        """
        n = len(w)
        F = []
        for k in range(self.nbs):
            F += [self.initial[k]*self.emissions[k, w[0]]]
        F = np.array(F)
        for i in range(1, n):
            F = (F @ self.transitions)*self.emissions[:, w[i]]
        return F.sum()

    def pbw(self, w):
        """

        :param w: séquence générée par le HMM self
        :return: la probabilité que self génère cette séquence
        """
        n = len(w)
        B = []
        for k in range(self.nbs):
            B += [[1]]
        B = np.array(B)
        for i in range(n-1, 0, -1):
            B = np.dot(self.transitions*self.emissions[:, w[i]], B)
        B = B*self.initial*self.emissions[:, w[0]]
        return B.sum()

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
                    probabilité_t = chemins[s][1]*self.emissions[(s, w[i])]*self.transitions[(s, k)]
                    if probabilité_t >= probabilité:
                        chemin = (chemins[s][0] + [s], probabilité_t)
                        probabilité = probabilité_t
                chemins_t.append(chemin)
            chemins = chemins_t

        return max(chemins, key=lambda x: x[1])

# Exo 13
    def predit(self,w):
        H = self.initial
        for i in range(1, len(w)):
            H = np.dot(self.transitions*self.emissions.T[:, w[i]], H)
        H = list(H)
        return H.index(max(H))
