import os, sys
#sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.." + "/.."))
#print sys.path
from os import listdir
import pyrebase
from cluster_space import Clusterspace
from add_data import DataAdder
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
firebase = pyrebase.initialize_app(config)
data_dir = "./Room1_preprocess"
space = Clusterspace(data_dir=data_dir)
D = DataAdder(data_dir=data_dir)

def count_cluster(cur_dir):
    count = set()
    for pic in listdir(cur_dir):
        if pic.split(".")[-1] == "jpg":
            name = pic.split(".")[0]
            clusterID, picID =name.split("_")
            count.add(clusterID)
    return len(count)





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
    D.add_data(cur_dir)
    space.train()
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
    if(frame["macs"] != None):
        for mac in frame["macs"]:
            f.write(mac + '\n')
            print "recived MAC:" + str(mac)
    f.close()
    if frame["links"] != None:
        for idx, link in enumerate(frame["links"]):
            imgur_download.getImg(cur_dir, "http://125.212.233.106:3000/" + link, idx)

        macs_result = match_mac_image(cur_dir)
        response = send_cv_result(idDevice, macs_result)  # send the result to firebase for rashberry to listen
        print (response)  # <Response [200]> is success, check firebase for result
        print ('frame id ', cur_dir)  # looking for entry with frame id on firebase for result




# match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest5")
# match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest6")
# match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest7")
# match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest8")
# match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest9")

# macs_result = match_mac_image("/home/vdvinh/FaceNet/openface/demos/71483431485500")
# print macs_result
# print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest0")
# print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest1")
# print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest2")
# print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest3")
#match_mac_image("/home/vdvinh/FaceNet/openface/demos/face_cluster/testid/2a9fb0f5-dcbd-41c5-90e5-ed3172efc8f4")
# db = firebase.database()
# my_stream = db.child("/upload").stream(stream_handler=stream_handle)
#
# print ('breakpoint')

if __name__ == "__main__":
    db = firebase.database()
    my_stream = db.child("/upload").stream(stream_handler=stream_handle)
    print ('breakpoint')

    # macs_result = match_mac_image("/home/vdvinh/FaceNet/openface/demos/71483431485500")
    # print macs_result
    # print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest0")
    # print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest1")
    # print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest2")
    # print match_mac_image("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/framtest3")
    #match_mac_image("/home/vdvinh/FaceNet/openface/demos/face_cluster/testid/2a9fb0f5-dcbd-41c5-90e5-ed3172efc8f4")
    #db = firebase.database()
    #my_stream = db.child("/upload").stream(stream_handler=stream_handle)

    # print ('breakpoint')