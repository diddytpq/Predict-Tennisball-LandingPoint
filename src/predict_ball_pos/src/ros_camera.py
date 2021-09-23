#!/usr/bin/env python

import numpy as np
import time
import roslib
import sys
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
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

roslib.load_manifest('ball_trajectory')

record = True

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/camera_left_top_ir/camera_left_top_ir/color/image_raw", Image, self.callback_left_top_ir)

        rospy.Subscriber("/camera_left_0/depth/image_raw",Image,self.callback_left_depth_0)
        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.callback_left_0)

        rospy.Subscriber("/camera_left_1/depth/image_raw",Image,self.callback_left_depth_1)
        rospy.Subscriber("/camera_left_1_ir/camera_left_1/color/image_raw",Image,self.callback_left_1)

        rospy.Subscriber("/camera_right_0/depth/image_raw",Image,self.callback_right_depth_0)
        rospy.Subscriber("/camera_right_0_ir/camera_right_0/color/image_raw",Image,self.callback_right_0)

        rospy.Subscriber("/camera_right_1/depth/image_raw",Image,self.callback_right_depth_1)
        rospy.Subscriber("/camera_right_1_ir/camera_right_1/color/image_raw",Image,self.callback_right_1)

        self.frame_recode = np.zeros([360,1280,3], np.uint8)

        




        if record == True:
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            self.out_0 = cv2.VideoWriter("train_video.mp4", self.codec, 35, (1280,640))
            #self.out_1 = cv2.VideoWriter("right.mp4", self.codec, 60, (640,640))

    def callback_left_top_ir(self, data):
        try:
            self.t0 = time.time()
            self.left_top_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)
     

        

    def callback_left_0(self, data):
        try:
            self.t0 = time.time()
            self.left_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_left_depth_0(self, data):
        try:
            depth_data =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            #self.left_depth_0 = np.uint8(depth_data) 
            self.left_depth_0 = np.uint8(np.round(depth_data))
            
            print(depth_data[223, 457])


        except CvBridgeError as e:
            print(e)

    def callback_left_1(self, data):
        try:
            self.left_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_left_depth_1(self, data):
        try:
            depth_data =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            #self.left_depth_1 = np.uint8(depth_data) 
            self.left_depth_1 = np.round(depth_data) 


        except CvBridgeError as e:
            print(e)

    def callback_right_0(self, data):
        try:
            self.right_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)

    def callback_right_depth_0(self, data):
        try:
            depth_data =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            self.right_depth_0 = np.uint8(depth_data) 

        except CvBridgeError as e:
            print(e)


    def callback_right_1(self, data):
        try:
            self.right_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

            self.main()
        except CvBridgeError as e:
            print(e)


    def callback_right_depth_1(self, data):
        try:
            depth_data =self.bridge.imgmsg_to_cv2(data, "passthrough")
            depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            self.right_depth_1 = np.uint8(depth_data ) 

        except CvBridgeError as e:
            print(e)


    def main(self):
        (rows,cols,channels) = self.right_data_1.shape

        if cols > 60 and rows > 60 :
            t1 = time.time()



            """self.left_data_0 = cv2.resize(self.left_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.left_data_1 = cv2.resize(self.left_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.right_data_0 = cv2.resize(self.right_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.right_data_1 = cv2.resize(self.right_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)"""

            #self.left_depth_0 = cv2.applyColorMap(self.left_depth_0, cv2.COLORMAP_JET)
            """self.left_depth_1 = cv2.applyColorMap(self.left_depth_1, cv2.COLORMAP_JET)
            self.right_depth_0 = cv2.applyColorMap(self.right_depth_0, cv2.COLORMAP_JET)
            self.right_depth_1 = cv2.applyColorMap(self.right_depth_1, cv2.COLORMAP_JET)"""

            self.left_frame = cv2.vconcat([self.left_data_0,self.right_data_0])
            self.right_frame = cv2.vconcat([self.left_data_1,self.right_data_1])
            self.left_top_frame = cv2.resize(self.left_top_data_0,(640,640),interpolation = cv2.INTER_AREA)

            self.main_frame = cv2.hconcat([self.left_frame,self.left_top_frame])
            
            print(self.left_frame.shape)
            print(self.left_top_frame.shape)

            #self.main_frame = cv.


            #self.frame_recode = self.main_frame

            t2 = time.time()
            cv2.imshow("left_frame", self.left_frame)
            #cv2.imshow("right_frame", self.right_frame)

            #cv2.imshow("left_data_0", self.left_data_0)
            #cv2.imshow("left_data_1", self.left_data_1)
            #cv2.imshow("right_data_0", self.right_data_0)
            #cv2.imshow("right_data_1", self.right_data_1)

            #cv2.imshow("left_depth_0", self.left_depth_0)
            #cv2.imshow("left_depth_1", self.left_depth_1)
            #cv2.imshow("right_depth_0", self.right_depth_0)
            #cv2.imshow("right_depth_1", self.right_depth_1)
            
            cv2.imshow("left_top_frame", self.main_frame)


            


            self.t1 = time.time()
            #print(self.t1 - self.t0)
            

            if record == True:
                self.out_0.write(self.main_frame)
                #self.out_1.write(self.right_frame)

            #print((t2-t1))
            key = cv2.waitKey(1)

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
