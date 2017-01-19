from os import listdir
import os
from munkres import Munkres
from face_pic import FacePic
from collections import OrderedDict, defaultdict
from face_cluster import FaceCluster
from cluster_space import Clusterspace
from Encoder import encoder
import pickle
import numpy as np

E = encoder()

def pad_to_square(a, pad_value=0):
    m = a.reshape((a.shape[0], -1))
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
    padded[0:m.shape[0], 0:m.shape[1]] = m
    return padded

def cls_cls_mactching(cla, clb, RECOGNIZE_THRESHOLD = 0.9):
    a = OrderedDict(sorted(cla.items(), key=lambda t: t[0]))
    b = OrderedDict(sorted(clb.items(), key=lambda t: t[0]))
    dist = []
    for i in a.keys():
        i_dist = []
        for j in b.keys():
            d = a[i].distance(b[j])
            if d > RECOGNIZE_THRESHOLD:
                d = 30041975
            i_dist.append(d)
        dist.append(i_dist)

    d = list(pad_to_square(np.asarray(dist)))
    m = Munkres()
    indexes = m.compute(d)
    ret = []
    for i, j in indexes:
        if d[i][j] == 30041975:
            continue
        if i < len(a.keys()) and j < len(b.keys()):
            ret.append((a.keys()[i],b.keys()[j]))
    return ret


def read_fram(f_path):
    framID = f_path.split("/")[-1]
    clusters = []
    for cluster in listdir(f_path):
        if os.path.isfile(f_path + "/" + cluster):
            continue
        face_cluster = set()
        for pic in listdir(f_path + "/" + cluster):
            path = f_path + "/" + cluster + "/" + pic
            face_cluster.add(FacePic(E.get_rep_preprocessed(path), framID, path))
        clusters.append(FaceCluster(face_cluster))

    f = open(f_path + "/" + "macs.txt")
    macs = []
    for line in f.readlines():
        mac = line.replace("\n", "")
        macs.append(mac)
    return clusters, macs


if __name__ == "__main__":
    a = read_fram("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testStore/testMoveData0")
    space = Clusterspace(data_dir=None)
    space.load_model()
    clusters, idx = space.getWorkingCluster()
    cla = defaultdict()
    clb = defaultdict()
    for i in idx:
        cla[i] = space.clusters_space[i]
    for i in range(len(a[0])):
        clb[i] = a[0][i]
    h = cls_cls_mactching(cla,clb)
    for i, j in h:
        space.clusters_space[i].show_faces()
        a[0][j].show_faces()
    print
