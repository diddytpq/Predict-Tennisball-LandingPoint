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

        self.image_left = rospy.Subscriber("/camera_left/image_raw",Image,self.callback_left)
        self.image_right = rospy.Subscriber("/camera_right/image_raw",Image,self.callback_right)


    def callback_left(self,data):
        try:
            self.left_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)


    def callback_right(self,data):
        try:
            self.right_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        (rows,cols,channels) = self.right_data.shape

        if cols > 60 and rows > 60 :
            t1 = time.time()

            self.left_frame = cv2.resize(self.left_data,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.right_frame = cv2.resize(self.right_data,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.main_frame = cv2.hconcat([self.left_frame,self.right_frame])

            t2 = time.time()
            cv2.imshow("main", self.main_frame)

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