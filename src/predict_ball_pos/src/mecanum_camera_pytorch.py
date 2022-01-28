#!/usr/bin/env python

from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])

import numpy as np
import time
import roslib
import rospy
import cv2
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import time
from pathlib import Path

import cv2

import os
import sys
import json
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import numpy as np
from torch_utils.dataloader_custom import TrackNetLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import math
from PIL import Image
import time
from torch_utils.network import *


roslib.load_manifest('ball_trajectory')

BATCH_SIZE = 1
HEIGHT=288
WIDTH=512

parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--video_name', type=str,
                    default='videos/test_gazebo.mp4', help='input video name for predict')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='weights/gazebo.tar', help='input model weight for predict')
parser.add_argument('--optimizer', type=str, default='Ada',
                    help='Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--record', type=bool, default=False,
                    help='record option')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())

def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
            torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)

def tran_input_img(img_list):

    trans_img = []

    for i in range(len(img_list)):

        img = img_list[i]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = cv2.resize(img,(WIDTH, HEIGHT))
        img = np.asarray(img).transpose(2, 0, 1) / 255.0

        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])

    trans_img = np.asarray(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])

def find_ball(pred_image, image_ori, ratio_w, ratio_h):

    if np.amax(pred_image) <= 0: #no ball
        return image_ori

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)
    # print(type(stats))

    if len(stats): 
        stats = np.delete(stats, 0, axis = 0)
        centroids = np.delete(centroids, 0, axis = 0)

    x, y , w, h, area = stats[np.argmax(stats[:,-1])]
    x_cen, y_cen = centroids[np.argmax(stats[:,-1])]

    cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
    cv2.circle(image_ori, (int(x_cen * ratio_w), int(y_cen * ratio_h)),  3, (0,0,255), -1)


    #for i in range(len(stats)):
    #    x, y, w, h, area = stats[i]

    return image_ori

model = efficientnet_b3()
model.to(device)
if args.optimizer == 'Ada':
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
checkpoint = torch.load(args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()

        self.input_img = []

        rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image,self.callback_depth)
        rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image,self.callback_camera)

        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.main)

        self.frame_recode = np.zeros([360,1280,3], np.uint8)

        if args.record == True:
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            #self.out_0 = cv2.VideoWriter("train_video.mp4", self.codec, 35, (1280,640))
            self.out_0 = cv2.VideoWriter("record_video.mp4", self.codec, 30, (640,480))

    def callback_camera(self, data):
        try:
            self.t0 = time.time()
            self.camera_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)
     
    def callback_depth(self, data):
        try:
            self.camera_depth_data = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.camera_depth = np.uint8(self.camera_depth_data) 


        except CvBridgeError as e:
            print(e)


    def ball_tracking(self, img):

        img = cv2.resize(img.copy(),(WIDTH, HEIGHT))


        self.input_img.append(img)

        if len(self.input_img) < 3:
            return 0

        if len(self.input_img) > 3:
            self.input_img = self.input_img[-3:]

        #unit = unit.reshape(1,9,unit.size()[-2],unit.size()[-1])
        t0 = time.time()

        unit = tran_input_img(self.input_img)

        unit = torch.from_numpy(unit).to(device, dtype=torch.float)
        torch.cuda.synchronize()
        
        with torch.no_grad():

            h_pred = model(unit)
            torch.cuda.synchronize()

            
            h_pred = (h_pred * 255).cpu().numpy()
            
            torch.cuda.synchronize()
            h_pred = (h_pred[0]).astype('uint8')
            h_pred = np.asarray(h_pred).transpose(1, 2, 0)
            #print(h_pred.shape)

            h_pred = (50 < h_pred) * h_pred

        #frame = find_ball(h_pred, frame, ratio_w, ratio_h)

        h_pred = cv2.resize(h_pred, dsize=(self.cols, self.rows), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)


        return h_pred
    

    def main(self, data):
        (self.rows, self.cols,channels) = self.camera_data.shape

        self.ratio_h = self.rows / HEIGHT
        self.ratio_w = self.cols / WIDTH

        if self.cols > 60 and self.rows > 60 :
            t1 = time.time()

            #self.camera_data = cv2.resize(self.camera_data,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            
            self.camera_depth = cv2.applyColorMap(self.camera_depth * 20, cv2.COLORMAP_JET)
            
            #self.main_frame = cv2.hconcat([self.camera_data,self.camera_depth])

            ball_image = self.ball_tracking(self.camera_data.copy()) 

            #robot_detect_img = self.robot_tracking(self.camera_data.copy()) #get robot bbox

            #self.frame_recode = self.main_frame

            t2 = time.time()

            #self.main_frame = cv2.hconcat([self.camera_data, robot_detect_img, ball_image])
            self.main_frame = cv2.hconcat([self.camera_data, self.camera_depth])

            cv2.imshow("main_frame", self.camera_data)

            cv2.imshow("main_frame", self.main_frame)
            #cv2.imshow("camera_frame", self.camera_data)
            #cv2.imshow("fgmask_dila", self.fgmask_dila)

            #cv2.imshow("ball_image", ball_image)
            

            print("FPS : ",1 / (t2 - t1))

            if args.record == True:
                self.out_0.write(self.camera_data)
                #self.out_0.write(self.main_frame)
                #self.out_1.write(self.right_frame)

            #print((t2-t1))
            key = cv2.waitKey(33)

            if key == 27 : 
                cv2.destroyAllWindows()
                return 0

            if key == ord("s"):
                cv2.imwrite("chess ({}).jpg".format(self.t1), self.left_data_0)





def main(args):

    ic = Image_converter()
    rospy.init_node('Image_converter', anonymous=True)
    
    try:
        rospy.spin()



    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
