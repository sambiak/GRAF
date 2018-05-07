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
            raise TypeError("nbl doit être entier")
        if nbl <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbl = nbl

    @property
    def nbs(self):
        return self.__nbs

    @nbs.setter
    def nbs(self, nbs):
        if not isinstance(nbs, int):
            raise TypeError("nbl doit être entier")
        if nbs <= 0:
            raise ValueError("nbl doit être strictement positif")
        self.__nbs = nbs

    @property
    def initial(self):
        return self.__initial

    @initial.setter
    def initial(self, initial):
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
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions):
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
        return self.__emissions

    @emissions.setter
    def emissions(self, emissions):
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
        return S, O

    def genere_f(self, w):
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
        F = []
        for k in range(self.nbs):
            F += [self.initial[k]*self.emissions[k, w[0]]]
        F = np.array(F)
        for i in range(1, n):
            F = (F  @ self.transitions)*self.emissions[:, w[i]]
        return F.sum()

    def genere_b(self, w):
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
                    if i < (len(w) - 1) :
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
        v = self.viterbi(w)
        res = 1
        for i in range(len(w)):
            res *= self.emissions[v[i], w[i]]
        return res

    def pviterbi_w(self, w):
        v = self.viterbi(w)
        p = self.pfw(w)
        pwv = self.initial[v[0]]
        for i in range(len(w)-1):
            pwv *= self.emissions[v[i], w[i]] * self.transitions[v[i], v[i+1]]
        pwv *= self.emissions[v[-1], w[-1]]
        return pwv/p

# Exo 16
    def nbviterbi(self, w, s):
        v = self.viterbi(w)
        nb = 0
        for i in range(len(w)):
            if v[i] == s[i]:
                nb += 1
        return nb

# Exo 13
    def predit(self, w):
        H = self.initial
        for i in range(len(w)):
            H = (self.transitions.T*self.emissions[:, w[i]].T) @ H
        P = []
        for l in range(self.nbl):
            P += [self.emissions[:, l] @ H]
        return P.index(max(P))

    def log_vraissemblance(self, S):
        res = 0
        for w in S:
            res += np.emath.log(self.pfw(w))
        return res

    @staticmethod
    def epsilon(m0, w, f, b):
        epsilon = []
        for t in range(len(w) - 1):
            dénominateur = np.einsum('k,kl,l,l->', f[:, t], m0.transitions, m0.emissions[:, w[t + 1]], b[:, t + 1])
            numérateur = f[:, t] * m0.transitions * (m0.emissions[:, w[t + 1]] * b[:, t + 1]).T
            epsilon.append(numérateur / dénominateur)
        return np.array(epsilon)

    @staticmethod
    def gamma(f, b):
        return (f * b) / np.einsum('ki,ki->i', f, b)

    @staticmethod
    def BW1(m0, s):
        """

        :type m0: HMM
        """
        pi = np.zeros(m0.nbs)
        T = np.zeros((m0.nbs, m0.nbs))
        O = np.zeros((m0.nbs, m0.nbl))
        for w in s:
            f = m0.genere_f(w)
            b = m0.genere_b(w)
            epsilon = HMM.epsilon(m0, w, f, b)
            gamma = HMM.gamma(f, b)
            pi += gamma[:,1]
            T += np.einsum('klt->kl', epsilon)
            for o in range(m0.nbl):
                for t in range(len(w)):
                    if w[t] == o:
                        O[:, o] += gamma[:, t]


        #Normalisation
        pi = pi/m0.nbs
        T = T/ np.einsum('kj->j', T)
        O = O/ np.einsum('ko -> o', O)

        return HMM(m0.nbl, m0.nbs, pi, T, O)

    @staticmethod
    def gen_HMM(nbs, nbl):
        rd.seed()
        sum = 0
        initial = []
        for i in range(nbs):
            x = rd.random()
            initial += [x]
            sum += x
        for i in range(nbs):
            initial[i] /= sum
        transitions = []
        for j in range(nbs):
            transitions += [[]]
            sum = 0
            for i in range(nbs):
                x = rd.random()
                transitions[j] += [x]
                sum += x
            for i in range(nbs):
                transitions[j][i] /= sum
        emissions = []
        for j in range(nbs):
            emissions += [[]]
            sum = 0
            for i in range(nbl):
                x = rd.random()
                emissions[j] += [x]
                sum += x
            for i in range(nbl):
                emissions[j][i] /= sum
        initial = np.array(initial)
        transitions = np.array(transitions)
        emissions = np.array(emissions)
        return HMM(nbl, nbs, initial, transitions, emissions)

    @staticmethod
    def BW2(nbs, nbl, S, N):
        M = HMM.gen_HMM(nbs, nbl)
        for i in range(N):
            M = HMM.BW1(M, S)
        return M

    @staticmethod
    def BW3(nbs, nbl, w, n, m):
        mlijhdfsk = [] #serieux ça fait quoi ce machin
        for i in range(m):
            mlijhdfsk += [HMM.BW2(nbs, nbl, [w], n)]
        return max(mlijhdfsk, key=lambda x: x.pfw(w))
