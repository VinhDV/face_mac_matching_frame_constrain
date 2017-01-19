from collections import defaultdict
from os import listdir, mkdir
import cv2
from preprocesser import preprocesser

class DataAdder:
    def __init__(self, data_dir):
        self.p = preprocesser()
        self.data_dir = data_dir

    def isValidFrame(self, fram_dir):
        framID = fram_dir.split("/")[-1]
        cluster2pic = defaultdict(lambda: defaultdict())
        macs = []

        for pic in listdir(fram_dir):
            if pic.split(".")[-1] == "jpg":
                name = pic.split(".")[0]
                clusterID, picID = name.split("_")
                preporcessed_face = self.p.preprocess(fram_dir + "/" + pic)
                if preporcessed_face != None:
                    cluster2pic[clusterID][picID] = preporcessed_face

            elif pic.split(".")[-1] == "txt":
                with open(fram_dir + "/" + pic) as f:
                    content = f.readlines()
                count = 0
                for mac in content:
                    if count > 0:
                        preporcessed_mac = mac.replace("\n", "").replace("\t", "")
                        if preporcessed_mac != "":
                            macs.append(preporcessed_mac)
                    count += 1
                f.close()

        count_total_num_perprocessed_pic = 0
        for cluster in cluster2pic.keys():
            count_total_num_perprocessed_pic += len(cluster2pic[cluster].keys())

        if len(macs) == 0 or count_total_num_perprocessed_pic == 0:
            return None, None

        return cluster2pic, macs

    def add_data(self, fram_dir):
        """
        :param fram_dir: fir of the frame to add
        :param data_dir: dir where all frames are stored
        :return: None
        """
        framID = fram_dir.split("/")[-1]
        cluster2pic, macs = self.isValidFrame(fram_dir)
        if cluster2pic == None:
            return

        new_path = self.data_dir + "/" + framID
        mkdir(new_path)

        for cluster in cluster2pic.keys():
            new_cluster = new_path + "/" + str(cluster)
            if len(cluster2pic[cluster]) > 0:
                mkdir(new_cluster)
                for pic in cluster2pic[cluster].keys():
                    cv2.imwrite(new_cluster + "/" + pic + ".jpg", cluster2pic[cluster][pic])

        with open(new_path + "/macs.txt", "w") as f:
            for mac in macs:
                f.write(str(mac) + "\n")
        f.close()


if __name__ == "__main__":
    D = DataAdder(data_dir="./testStore")
    D.add_data("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/testMoveData0")
    D.add_data("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/testMoveData1")
    D.add_data("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/testMoveData2")
    D.add_data("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/testMoveData3")
    D.add_data("/media/vdvinh/newdisk/FaceNet/openface/demos/face_cluster/testid/testMoveData4")
