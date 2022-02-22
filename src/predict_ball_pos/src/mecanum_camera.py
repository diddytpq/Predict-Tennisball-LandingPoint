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

record = True

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image,self.callback_depth)
        rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image,self.callback_camera)

        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.main)

        if record == True:
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            self.out_0 = cv2.VideoWriter("record_video.mp4", self.codec, 30, (1280,720))

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


    def main(self, data):
        (rows,cols,channels) = self.camera_data.shape

        if cols > 60 and rows > 60 :
            t1 = time.time()
            
            self.camera_depth = cv2.applyColorMap(self.camera_depth * 20, cv2.COLORMAP_JET)
            
            t2 = time.time()

            self.main_frame = cv2.hconcat([self.camera_data, self.camera_depth])

            cv2.imshow("main_frame", self.main_frame)

            print("FPS : ",1 / (t2 - t1))

            if record == True:

                self.out_0.write(self.main_frame)

            key = cv2.waitKey(33)

            if key == 27 : 
                cv2.destroyAllWindows()
                return 0


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
