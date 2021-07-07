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



roslib.load_manifest('ball_trajectory')



class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        rospy.init_node('Image_converter', anonymous=True)

        self.image_left_0 = rospy.Subscriber("/camera_left_0/image_raw",Image,self.callback_left_0)
        self.image_left_1 = rospy.Subscriber("/camera_left_1/image_raw",Image,self.callback_left_1)
        self.image_right_0 = rospy.Subscriber("/camera_right_0/image_raw",Image,self.callback_right_0)
        self.image_right_1 = rospy.Subscriber("/camera_right_1/image_raw",Image,self.callback_right_1)

    def callback_left_0(self,data):
        try:
            self.left_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_left_1(self,data):
        try:
            self.left_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_right_0(self,data):
        try:
            self.right_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)

    def callback_right_1(self,data):
        try:
            self.right_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)

        (rows,cols,channels) = self.right_data_1.shape

        if cols > 60 and rows > 60 :
            t1 = time.time()

            self.left_frame_0 = cv2.resize(self.left_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.left_frame_1 = cv2.resize(self.left_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.right_frame_0 = cv2.resize(self.right_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.right_frame_1 = cv2.resize(self.right_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.left_frame = cv2.hconcat([self.left_frame_0,self.left_frame_1])
            self.right_frame = cv2.hconcat([self.right_frame_0,self.right_frame_1])


            t2 = time.time()
            cv2.imshow("left_frame", self.left_frame)
            cv2.imshow("right_frame", self.right_frame)


            print((t2-t1))
            key = cv2.waitKey(1)

            if key == 27 : 
                cv2.destroyAllWindows()

                return 0



def main(args):

    ic = Image_converter()
    
    try:
        rospy.spin()



    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)