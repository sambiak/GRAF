import numpy as np
import random as rd


class HMM():
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
        if np.shape(initial) != (1, self.nbs):
            raise ValueError("initial n'a pas la bonne dimension")
        if not np.isclose(np.array([initial.np.sum()]), np.array([1.0])):
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
            raise ValueError("transitions n'a pas la bonne dimension")
        for line in range(self.nbs):
            if not np.isclose(np.array([transitions[line].np.sum()]), np.array([1.0])):
                raise ValueError("la somme des probabilités de transition, ligne par ligne, doit être 1")
        for t in transitions:
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
            if not np.isclose(np.array([emissions[line].np.sum()]), np.array([1.0])):
                raise ValueError("la somme des probabilités de transition, ligne par ligne, doit être 1")
        for e in emissions:
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

    @staticmethod
    def load(adr):
        lines = HMM.__ligns_not_comments(adr)
        nbl = int(next(lines))
        nbs = int(next(lines))
        initial = [int(next(lines)) for i in range(nbs)]
        transitions = [[j for j in map(int, next(lines).split())] for i in range(nbs)]
        emmissions = [[j for j in map(int, next(lines).split())] for i in range(nbl)]


    def save(self,adr='HMM.txt'):
        fichier = open(adr, "w")
        fichier.write("# The number of letters \n")
        fichier.write(str(self.nbl))
        fichier.write("# The number of states \n")
        fichier.write(str(self.nbs))
        fichier.write("\n # The initial transitions \n")
        for l in self.initial :
            fichier.write('i'+"\n")
        fichier.write("\n # The internal transitions \n")
        for l in self.transitions:
            fichier.write('i' + "\n")
        fichier.write("\n # The emissions\n")
        for l in self.emissions :
            fichier.write('i'+"\n")
        fichier.close()


    @staticmethod
    def draw_multinomial(L):
        x = rd.random()
        L = [0] + L
        for i in range(len(L)-1):
            if L[i] <= x <= L[i+1]:
                return i+1

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
        n = len(w)
        F = []
        for k in range(self.nbs):
            F += [self.initial[k]*self.emissions[k, w[0]]]
        F = np.array(F)
        for i in range(1, n):
            F = np.dot(F, self.transitions)*self.emissions[:, w[i]]
        return F.sum()
