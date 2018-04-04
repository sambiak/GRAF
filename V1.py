import numpy as np


class HMM():
    """ Define an HMM"""

    def __init__(self, nbl, nbs, initial, transitions, emissions):
        if not isinstance(emissions, np.ndarray):
            raise ValueError("emissions doit être un array numpy")
        if np.shape(emissions) != (nbs, nbl):
            raise ValueError("emissions n'a pas la bonne dimension")
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
