import unittest
from V1 import HMM
import numpy as np


class TestSave(unittest.TestCase):

    def test_save1(self):
        t1 = HMM.load("save1.txt")
        t1.save()

    def test_draw_multinomial(self):
        self.assertEqual(0, HMM.draw_multinomial(np.array([1, 0, 0])))

    def test_gen_rand(self):
        M = HMM.load('test_gen_rand')
        self.assertEqual([0, 1, 2, 0, 1, 2, 0, 1, 2], M.gen_rand(9))

    def test_pfw(self):
        M = HMM.load('test_gen_rand')
        self.assertEqual(1, M.pfw([0, 1, 2]))

    def test_pbw(self):
        M = HMM.load('test_gen_rand')
        self.assertEqual(1, M.pbw([0, 1, 2]))

if __name__ == '__main__':
    unittest.main()
