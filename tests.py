import unittest
from V1 import HMM
import numpy as np


class TestHMM(unittest.TestCase):

    def test_save1(self):
        t1 = HMM.load("save1.txt")
        t1.save()
    ####################################################################
    # Test Multinomial
    ####################################################################

    def test_draw_multinomial(self):
        self.assertEqual(0, HMM.draw_multinomial(np.array([1, 0, 0])))
    ####################################################################
    # Test Rand
    ####################################################################

    def test_gen_rand(self):
        M = HMM.load('save3.txt')
        self.assertEqual([0, 1, 2, 0, 1, 2, 0, 1, 2], M.gen_rand(9))

    ####################################################################
    # Test PFW
    ####################################################################

    def test_pfw(self):
        M = HMM.load('save3.txt')
        self.assertEqual(1, M.pfw([0, 1, 2]))

    def test_pfw2(self):
        M = HMM.load('save2.txt')
        self.assertEqual(0.5, M.pfw([0, 1, 2]))

    ####################################################################
    # Test PBW
    ####################################################################

    def test_pbw(self):
        M = HMM.load('save3.txt')
        self.assertEqual(1, M.pbw([0, 1, 2]))

    def test_pbw2(self):
        M = HMM.load('save2.txt')
        self.assertEqual(0.5, M.pbw([0, 1, 2]))

    ####################################################################
    # Test Predit
    ####################################################################

    def test_predit(self):
        M = HMM.load('save3.txt')
        self.assertEqual(2, M.predit([0, 1]))

    ####################################################################
    # Test Viterbi
    ####################################################################

    def test_viterbi(self):
        M = HMM.load('save3.txt')
        self.assertEqual(([0, 1, 2], 1.0), M.viterbi([0, 1, 2]))

    def test_viterbi2(self):
        M = HMM.load('save2.txt')
        self.assertEqual(([0, 1, 2], 0.5), M.viterbi([0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
