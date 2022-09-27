import cv2
import numpy as np
import os
import json

class general_pose_model(object):
    def __init__(self, modelpath):
        # Specify the model to be used
        #   Body25: 25 points
        #   COCO:   18 points
        #   MPI:    15 points
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.05
        self.pose_net = self.general_coco_model(modelpath)

    def general_coco_model(self, modelpath):
        self.points_name = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "REye": 14, "LEye": 15,
            "REar": 16, "LEar": 17,
            "Background": 18}
        self.num_points = 18
        self.point_pairs = [[1, 0], [1, 2], [1, 5],
                            [2, 3], [3, 4], [5, 6],
                            [6, 7], [1, 8], [8, 9],
                            [9, 10], [1, 11], [11, 12],
                            [12, 13], [0, 14], [0, 15],
                            [14, 16], [15, 17]]
        prototxt   = os.path.join(
            modelpath,
            'pose_deploy_linevec.prototxt')
        caffemodel = os.path.join(
            modelpath,
            'pose_iter_440000.caffemodel')
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        inpBlob = cv2.dnn.blobFromImage(img_cv2,
                                        1.0 / 255,
                                        (self.inWidth, self.inHeight),
                                        (0, 0, 0),
                                        swapRB=False,
                                        crop=False)
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :] # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append(x)
                points.append(y)
                points.append(prob)
            else:
                points.append(0)
                points.append(0)
                points.append(0)

        return points

def generate_pose_keypoints(img_file, pose_file):

    modelpath = 'ACGPN/pose'
    pose_model = general_pose_model(modelpath)

    res_points = pose_model.predict(img_file)

    pose_data = {"version": 1,
                 "people":  [
                                {"pose_keypoints": res_points}
                            ]
                }

    pose_keypoints_path = pose_file

    json_object = json.dumps(pose_data, indent = 4)

    # Writing to sample.json
    with open(pose_keypoints_path, "w") as outfile:
        outfile.write(json_object)
    print('File saved at {}'.format(pose_keypoints_path))

