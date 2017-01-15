import os
import sys
import unittest

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from face_cluster import FaceCluster
from face_pic import FacePic
from cluster_space import Clusterspace
import numpy as np


class TestClusterSpace(unittest.TestCase):
    def testFacePic(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f1', None)
        p3 = FacePic(np.array([4, 4]), 'f2', None)
        p4 = FacePic(np.array([8, 8]), 'f2', None)
        p5 = FacePic(np.array([12, 12]), 'f2', None)
        c1 = FaceCluster(set([p1, p2]))
        c4 = FaceCluster(set([p1]))
        self.assertEqual(len(c1.facepics), 2)
        self.assertEqual(len(c4.facepics), 1)
        c2 = FaceCluster(set([p3, p4, p5]))
        self.assertEqual(len(c2.facepics), 3)
        self.assertTrue((c1.mean() == np.array([2., 2.])).all())
        self.assertTrue((c2.mean() == np.array([8., 8.])).all())
        self.assertTrue((c4.mean() == np.array([1., 1.])).all())
        d12 = c1.distance(c2)
        self.assertAlmostEqual(d12, 13.145341380123986)
        d21 = c2.distance(c1)
        self.assertAlmostEqual(d21, d12)
        d42 = c2.distance(c4)
        self.assertAlmostEqual(d42, 12.124355652982141)

    def testClusterSpace(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f1', None)
        p3 = FacePic(np.array([4, 4]), 'f4', None)
        p4 = FacePic(np.array([8, 8]), 'f2', None)
        p5 = FacePic(np.array([12, 12]), 'f2', None)
        c0 = FaceCluster(set([p1, p2]))
        c1 = FaceCluster(set([p3]))
        c2 = FaceCluster(set([p4, p5]))
        space = Clusterspace(None, [c0, c1, c2])
        self.assertEqual(len(space.clusters_space), 3)
        self.assertTrue(c0 in space.clusters_space and c1 in space.clusters_space and c2 in space.clusters_space)
        space.calculate_distance()
        self.assertAlmostEqual(c0.distance(c1), space.distances[0, 1])
        self.assertAlmostEqual(c1.distance(c2), space.distances[1, 2])
        self.assertAlmostEqual(c0.distance(c2), space.distances[0, 2])

    def testClusterSpace2(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f2', None)
        p3 = FacePic(np.array([4, 4]), 'f3', None)
        p4 = FacePic(np.array([8, 8]), 'f4', None)
        p5 = FacePic(np.array([12, 12]), 'f5', None)
        p6 = FacePic(np.array([10, 10]), 'f6', None)
        p7 = FacePic(np.array([15, 15]), 'f7', None)
        p8 = FacePic(np.array([22, 23]), 'f8', None)
        c0 = FaceCluster(set([p1, p2]))
        c1 = FaceCluster(set([p3]))
        c2 = FaceCluster(set([p4, p5]))
        c3 = FaceCluster(set([p6, p7, p8]))
        space = Clusterspace(None, [c0, c1, c2])
        space.calculate_distance()
        self.assertAlmostEqual(c0.distance(c1), space.distances[0, 1])
        self.assertAlmostEqual(c1.distance(c2), space.distances[1, 2])
        self.assertAlmostEqual(c0.distance(c2), space.distances[0, 2])
        space.add_clusters([c3])
        self.assertAlmostEqual(c3.distance(c0), space.distances[0, 3])
        self.assertAlmostEqual(c3.distance(c1), space.distances[1, 3])
        self.assertAlmostEqual(c3.distance(c2), space.distances[2, 3])

    def testMergeAble(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f1', None)
        p3 = FacePic(np.array([4, 4]), 'f1', None)
        p4 = FacePic(np.array([8, 8]), 'f1', None)
        p9 = FacePic(np.array([21, 25]), 'f1', None)
        p10 = FacePic(np.array([31, 25]), 'f1', None)
        p11 = FacePic(np.array([41, 25]), 'f1', None)
        p12 = FacePic(np.array([51, 25]), 'f1', None)
        p5 = FacePic(np.array([12, 12]), 'f2', None)
        p6 = FacePic(np.array([10, 10]), 'f2', None)
        p7 = FacePic(np.array([15, 15]), 'f2', None)
        p8 = FacePic(np.array([22, 23]), 'f2', None)
        p13 = FacePic(np.array([22, 33]), 'f2', None)
        pf3_1 = FacePic(np.array([100, 101]), 'f3', None)
        pf3_2 = FacePic(np.array([101, 101]), 'f3', None)
        pf3_3 = FacePic(np.array([102, 101]), 'f3', None)
        pf3_4 = FacePic(np.array([103, 101]), 'f3', None)
        pf3_5 = FacePic(np.array([104, 101]), 'f3', None)
        pf3_6 = FacePic(np.array([105, 101]), 'f3', None)

        c0 = FaceCluster(set([p1, p2]))  # f1 = 2
        c1 = FaceCluster(set([p3]))  # f1 = 3
        c3 = FaceCluster(set([p6, p7, p8]))  # f2 = 3
        c4 = FaceCluster(set([p1, p2, p3, p4]))  # f2 = 4
        c5 = FaceCluster(set([p5, p6, p7, p8, p9, p13]))  # f1 =1 ,f2 = 5
        c6 = FaceCluster(set([p5, p6, p7, p8, p9, p10, p11, p12]))  # f2 = 4, f1 = 4
        c7 = FaceCluster(set([pf3_1, pf3_2, pf3_3, pf3_4, pf3_5, pf3_6, p1]))  # f1 = 1, f3 = 6
        self.assertFalse(c0.is_mergeable(c1))
        self.assertTrue(c3.is_mergeable(c1))
        self.assertFalse(c4.is_mergeable(c5))
        self.assertFalse(c4.is_mergeable(c6))
        self.assertFalse(c7.is_mergeable(c1))
        self.assertTrue(c7.is_mergeable(c5))

    def test_split(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f2', None)
        p3 = FacePic(np.array([4, 4]), 'f3', None)
        p4 = FacePic(np.array([8, 8]), 'f1', None)
        c0 = FaceCluster(set([p1, p2, p3, p4]))
        child = c0.split(['f2', 'f3'])
        self.assertTrue(p1 in c0.facepics)
        self.assertTrue(not p2 in c0.facepics)
        self.assertTrue(not p3 in c0.facepics)
        self.assertTrue(p4 in c0.facepics)
        self.assertTrue(not (p1 in child.facepics))
        self.assertTrue(p2 in child.facepics)
        self.assertTrue(p3 in child.facepics)
        self.assertTrue(not (p4 in child.facepics))

    def test_clusterspace(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p2 = FacePic(np.array([3, 3]), 'f1', None)
        p3 = FacePic(np.array([4, 4]), 'f1', None)
        p4 = FacePic(np.array([8, 8]), 'f1', None)
        p9 = FacePic(np.array([21, 25]), 'f1', None)
        p10 = FacePic(np.array([31, 25]), 'f1', None)
        p11 = FacePic(np.array([41, 25]), 'f1', None)
        p12 = FacePic(np.array([51, 25]), 'f1', None)
        p5 = FacePic(np.array([12, 12]), 'f2', None)
        p6 = FacePic(np.array([10, 10]), 'f2', None)
        p7 = FacePic(np.array([15, 15]), 'f2', None)
        p8 = FacePic(np.array([22, 23]), 'f2', None)
        p13 = FacePic(np.array([22, 33]), 'f2', None)
        pf3_1 = FacePic(np.array([100, 101]), 'f3', None)
        pf3_2 = FacePic(np.array([101, 101]), 'f3', None)
        pf3_3 = FacePic(np.array([102, 101]), 'f3', None)
        pf3_4 = FacePic(np.array([103, 101]), 'f3', None)
        pf3_5 = FacePic(np.array([104, 101]), 'f3', None)
        pf3_6 = FacePic(np.array([105, 101]), 'f3', None)

        c0 = FaceCluster(set([p1, p2]))  # f1 = 2
        c1 = FaceCluster(set([p3]))  # f1 = 3
        c2 = FaceCluster(set([p4]))
        c3 = FaceCluster(set([p6, p7, p8]))  # f2 = 3
        c4 = FaceCluster(set([p1, p2, p3, p4]))  # f2 = 4
        c5 = FaceCluster(set([p5, p6, p7, p8, p9, p13]))  # f1 =1 ,f2 = 5
        c6 = FaceCluster(set([p5, p6, p7, p8, p9, p10, p11, p12]))  # f2 = 4, f1 = 4
        c7 = FaceCluster(set([pf3_1, pf3_2, pf3_3, pf3_4, pf3_5, pf3_6, p1]))  # f1 = 1, f3 = 6
        clusters = [c0, c1, c2, c3, c4, c5, c6, c7]
        clusterspace = Clusterspace(None, clusters)
        clusterspace.calculate_distance()

        for idx_a, c_a in enumerate(clusters):
            for idx_b, c_b in enumerate(clusters):
                if (idx_a < idx_b):
                    if (c_a.is_mergeable(c_b)):
                        distance = c_a.distance(c_b)
                        assert distance == clusterspace.distances[idx_a, idx_b]
                else:
                    assert not (idx_a, idx_b) in clusterspace.distances.keys()

    def test_merge(self):
        point1 = []

        root1 = (1, 1)
        root = root1
        for i in range(root[0], 3 + root[0]):
            for j in range(root[1], 3 + root[1]):
                point1.append((i, j))

        point2 = []
        root2 = (6, 1)
        root = root2

        for i in range(root[0], 3 + root[0]):
            for j in range(root[1], 3 + root[1]):
                point2.append((i, j))

        point3 = []
        root3 = (4, 5)
        root = root3

        for i in range(root[0], 3 + root[0]):
            for j in range(root[1], 3 + root[1]):
                point3.append((i, j))

        point = []
        point = point1 + point2 + point3

        faces = []
        clusters = []
        for idx, p in enumerate(point):
            f = FacePic(np.array(list(p)), str(idx), "p")
            c = FaceCluster(set([f]))
            # faces.append(f)
            clusters.append(c)

        clusterspace = Clusterspace(None, clusters)
        clusterspace.merge_closest(4)
        working_cluster, _ = clusterspace.getWorkingCluster()
        self.assertEqual(len(working_cluster), 3)

    def test_split_merge(self):
        p1 = FacePic(np.array([1, 1]), 'f1', None)
        p9 = FacePic(np.array([21, 25]), 'f1', None)
        p5 = FacePic(np.array([12, 12]), 'f2', None)
        p6 = FacePic(np.array([10, 10]), 'f2', None)
        p7 = FacePic(np.array([15, 15]), 'f2', None)
        p8 = FacePic(np.array([22, 23]), 'f2', None)
        p13 = FacePic(np.array([22, 33]), 'f2', None)
        pf3_1 = FacePic(np.array([100, 101]), 'f3', None)
        pf3_2 = FacePic(np.array([101, 101]), 'f3', None)
        pf3_3 = FacePic(np.array([102, 101]), 'f3', None)
        pf3_4 = FacePic(np.array([103, 101]), 'f3', None)
        pf3_5 = FacePic(np.array([104, 101]), 'f3', None)
        pf3_6 = FacePic(np.array([105, 101]), 'f3', None)

        c5 = FaceCluster(set([p5, p6, p7, p8, p9, p13]))  # f1 =1 ,f2 = 5
        c7 = FaceCluster(set([pf3_1, pf3_2, pf3_3, pf3_4, pf3_5, pf3_6, p1]))  # f1 = 1, f3 = 6
        ret, child, a_child = c5.merge(c7, split_child=True)
        self.assertTrue(p1 in a_child.facepics)
        self.assertTrue(p9 in child.facepics)
        self.assertTrue(not p1 in ret.facepics)
        self.assertTrue(not p9 in ret.facepics)


if __name__ == '__main__':
    unittest.main()
