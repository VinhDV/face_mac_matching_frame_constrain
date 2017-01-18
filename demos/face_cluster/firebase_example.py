import os
import re
from os import listdir
from preprocesser import preprocesser
import pyrebase
from cluster_space import Clusterspace
from collections import defaultdict
import imgur_download
from request_to_server import *
import cv2
config = {
    "apiKey": "AIzaSyBNeJcW6CKp9v7z9HjRJHcOj7Sw7qY14AM",
    "authDomain": "api-server-koa.firebaseapp.com",
    "databaseURL": "https://api-server-koa.firebaseio.com",
    "storageBucket": "api-server-koa.appspot.com",
    "messagingSenderId": "409014105383"
}
#firebase = pyrebase.initialize_app(config)
p = preprocesser()
data_dir = "./Room1_preprocess"
space = Clusterspace(data_dir=data_dir)

def count_cluster(cur_dir):
    count = set()
    for pic in listdir(cur_dir):
        if pic.split(".")[-1] == "jpg":
            name = pic.split(".")[0]
            clusterID, picID =name.split("_")
            count.add(clusterID)
    return len(count)



def add_data(fram_dir,data_dir):
    """
    :param fram_dir: fir of the frame to add
    :param data_dir: dir where all frames are stored
    :return: None
    """
    framID = fram_dir.split("/")[-1]
    cluster2pic = defaultdict(lambda :defaultdict())
    for pic in listdir(fram_dir):
        if pic.split(".")[-1] == "jpg":
            name = pic.split(".")[0]
            clusterID, picID =name.split("_")
            cluster2pic[clusterID][picID] = fram_dir+"/"+pic
    new_path = data_dir + "/" + framID
    os.mkdir(new_path)
    for cluster in cluster2pic.keys():
        new_cluster = new_path + "/" + str(cluster)
        os.mkdir(new_cluster)
        for pic in cluster2pic[cluster].keys():
            face = p.preprocess(cluster2pic[cluster][pic])
            if face != None:
                cv2.imwrite(new_cluster+"/"+pic+".jpg",face)


def match_mac_image(cur_dir):
    '''
       process cv clustering here
       :param cur_dir: directory
               macs.txt
               1.png
               2.png
       :return:
       an array of macs
       macs[0] is mac address of 0.png in cur_dir
       '''
    space.train()
    add_data(cur_dir, data_dir)
    FaceToName = space.match(cur_dir)

    rep = []
    if FaceToName == None:
        for i in range(count_cluster(cur_dir)):
            rep.append(0)
        return rep
    for i in range(len(FaceToName.keys())):
        rep.append(FaceToName[i])
    return rep


def stream_handle(message):
    '''
    will be called when something new on firebase
    will be called when api/detect was called
    :param message:
    :return:
    '''
    print ('Stream handle')
    data = message["data"]
    index = message["path"]
    if (index == '/'):  # skip the first one (which is none or old data)
        return
    index = index[1:]  # remove first character
    print (index)
    print (data)

    frame = data
    cur_dir = str(index)
    print ('frame id ',cur_dir)
    os.makedirs(os.path.abspath(cur_dir))  # create new folder

    f = open(cur_dir + '/' + 'macs.txt', 'w')

    # write id device at top of macs file
    idDevice = index.split('/')[0]
    f.write(idDevice + '\n')  # write idDevice on top of macs.txt
    for mac in frame["macs"]:
        f.write(mac + '\n')
        print (mac)
    f.close()
    for idx, link in enumerate(frame["links"]):
        imgur_download.getImg(cur_dir, "http://api-server-koa.herokuapp.com/" + link, idx)

    macs_result = match_mac_image(cur_dir)
    response = send_cv_result(idDevice, macs_result)  # send the result to firebase for rashberry to listen
    print (response)  # <Response [200]> is success, check firebase for result
    print ('frame id ', cur_dir)  # looking for entry with frame id on firebase for result


# macs_result = match_mac_image("/home/vdvinh/FaceNet/openface/demos/71483431485500")
# print macs_result
print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest0")
print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest1")
print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest2")
print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest3")
#match_mac_image("/home/vdvinh/FaceNet/openface/demos/face_cluster/testid/2a9fb0f5-dcbd-41c5-90e5-ed3172efc8f4")
#db = firebase.database()
#my_stream = db.child("/upload").stream(stream_handler=stream_handle)

print ('breakpoint')
