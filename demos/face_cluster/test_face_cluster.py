#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()
from os import listdir
import argparse
import cv2
import os

import numpy as np

np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRepPreprocessed(imgPath):
    bgrImg = cv2.imread(imgPath)
    return net.forward(bgrImg)
import dill
from collections import defaultdict
FramePath = args.imgs[0]
# data = defaultdict(lambda :defaultdict(list))
# path2rep = defaultdict(lambda :0)
# for frame in listdir(FramePath):
#     for cluster in listdir(FramePath + "/" + frame):
#         for pic in listdir(FramePath + "/" + frame + "/" + cluster):
#             path = FramePath + "/" + frame + "/" + cluster + "/" + pic
#             data[frame][cluster].append(path)
#             path2rep[path] = getRepPreprocessed(path)
import pickle
# pickle.dump(data, open("data.p", "wb"), protocol=2)
# pickle.dump(path2rep, open( "path2rep.p", "wb" ), protocol=2)
#data = pickle.load(open("data.p", "rb"))
#path2rep = pickle.load(open("path2rep.p", "rb"))

from face_cluster import FaceCluster
from face_pic import FacePic
from cluster_space import Clusterspace
# clusters = []
# for frame in data.keys():
#     for cluster in data[frame].keys():
#         FacePics = set()
#         for path in data[frame][cluster]:
#             FacePics.add(FacePic(path2rep[path],frame,path))
#         clusters.append(FaceCluster(FacePics))
# space = Clusterspace(clusters)
# space.merge_closest(2.0)
#pickle.dump(space, open("space.p", "wb"), protocol=2)
space = pickle.load(open("space.p", "rb"))
#space.show__working_cluster()
#space.show_working_cluster_and_distance()
test_dir = "/home/vdvinh/FaceNet/openface/demos/face_cluster/testcase"
for testfram in listdir(test_dir):
    space.match(test_dir+"/"+testfram)
# print
