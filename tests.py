import unittest
from V1 import HMM

class TestSave(unittest.TestCase):

    def test_save1(self):
        t1 = HMM.load("save1.txt")
        t1.save()

if __name__ == '__main__':
    unittest.main()
