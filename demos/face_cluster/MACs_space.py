from collections import defaultdict
from os import listdir
import os
from face_pic import FacePic
from face_cluster import FaceCluster
from cluster_space import Clusterspace
from munkres import Munkres
import util
import numpy as np

class MacsSpace:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mac_2_frams_path = defaultdict(list)
        self.mac_2_cluster_space = defaultdict()



    def match(self, f_path):
        working_macs = self.get_working_mac()
        if len(working_macs) == 0:
            return None
        f_path = self.data_dir + "/" + f_path.split("/")[-1]
        clusters, macs = util.read_fram(f_path)
        potential_mac = list(set(working_macs) & set(macs))
        if len(potential_mac) == 0:
            return None


        working_cluster_idx = working_macs


        dist = np.zeros((len(clusters), len(working_cluster_idx)))

        working_cluster_idx2idx = defaultdict()
        idx2working_cluster_idx = defaultdict(lambda: -1)

        count = 0
        for cluster_idx in working_cluster_idx:
            working_cluster_idx2idx[cluster_idx] = count
            idx2working_cluster_idx[count] = cluster_idx
            count += 1

        for acluster in working_cluster_idx:
            for cluster in range(len(clusters)):
                dist[cluster][working_cluster_idx2idx[acluster]] \
                    = (self.clusters_space[acluster]).distance(clusters[cluster])

        for i in range(dist.shape[0]):
            if np.all(dist[i] > RECOGNISE_THRESHOLD):
                for j in range(dist.shape[1]):
                    dist[i][j] = 987654321

        # for i in range(len(clusters)):
        #     a = clusters[i]
        #     a.show_faces()

        m = Munkres()
        dist = pad_to_square(dist, pad_value=0)
        tmp = dist.tolist()
        indexes = m.compute(tmp)

        ret = defaultdict()
        print "-------------------------------MATCH-----------------------------------------"
        for row, column in indexes:
            if tmp[row][column] == 987654321:
                idx2working_cluster_idx[row] = -1
            if tmp[row][column] != 0:
                ret[row] = self.clusters_space[idx2working_cluster_idx[column]].name
                print "\t" + str(row) + " : " + \
                      self.clusters_space[idx2working_cluster_idx[column]].name \
                      + " (dist: " + str(tmp[row][column]) + ")"

        # for i in ret.keys():
        #     acluster = clusters[i]
        #     cluster = self.clusters_space[ret[i]]
        #     acluster.show_faces()
        #     cluster.show_faces()

        return ret

    def train(self):
        self.update_mac_2_frams_path()
        for mac in self.mac_2_cluster_space.keys():
            self.mac_2_cluster_space[mac] = Clusterspace(data_dir=self.mac_2_frams_path[mac], mac=mac)
            self.mac_2_cluster_space[mac].find_host_cluster()

    def update_mac_2_frams_path(self):
        self.mac_2_frams_path = defaultdict(list)
        for fram in listdir(self.data_dir):
            f = open(self.data_dir + "/" + fram + "/" + "macs.txt")
            for line in f.readlines():
                mac = line.replace("\n","")
                self.mac_2_frams_path[mac].append(self.data_dir + "/" + fram)

    def get_working_mac(self):
        working_macs = []
        for mac in self.mac_2_cluster_space.keys():
            if self.mac_2_cluster_space[mac].is_matchable():
                working_macs.append(mac)
        return working_macs
if __name__ == "__main__":
    a = MacsSpace(data_dir="./testStore")
    a.update_mac_2_frams_path()
    print