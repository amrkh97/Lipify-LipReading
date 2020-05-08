import unittest
import sys
sys.path.append('../')
import project_prototype


class PrototypeTestCase(unittest.TestCase):
    def test_Project_Prototype(self):
        result = "again blue"
        videosPath = "../Prototype-Test-Videos/*.mp4"
        self.assertEqual(project_prototype.prototypeProject(videosPath), result)

    def test_NoVideos(self):
        result = "Error! No videos were passed"
        videosPath = "../Prototype-Test-Videos/"
        self.assertEqual(project_prototype.prototypeProject(videosPath), result)


if __name__ == '__main__':
    unittest.main()
