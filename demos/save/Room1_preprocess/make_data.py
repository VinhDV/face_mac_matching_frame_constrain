from os import listdir
from collections import defaultdict
import os
from random import shuffle, randrange
dir_path = os.path.dirname(os.path.realpath(__file__))
user2cluster = defaultdict(list)
for user in listdir(dir_path):
    if user == "make_data.py" or user == ".idea":
        continue
    for cluster in listdir(user):
        user2cluster[user].append(dir_path+"/"+user+"/"+cluster)
for user in listdir(dir_path):
    if user == "make_data.py" or user == ".idea":
        continue
    shuffle(user2cluster[user])

