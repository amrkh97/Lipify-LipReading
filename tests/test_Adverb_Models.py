import unittest
import sys
sys.path.append('../')
import Compare_Adverb_Models as AdverbModels


class AdverbTestCases(unittest.TestCase):
    def test_Adverb_Accuracies(self):
        resultAdverb = {'Custom_Arch': 92.78, 'VGG': 28.94, 'Numpy': 45.0}
        self.assertEqual(AdverbModels.validateAdverbModels(), resultAdverb)


if __name__ == '__main__':
    unittest.main()
