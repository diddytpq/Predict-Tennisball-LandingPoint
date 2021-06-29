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

fgbg = cv2.createBackgroundSubtractorMOG2(100, 16, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))





class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        rospy.init_node('Image_converter', anonymous=True)

        self.image_left = rospy.Subscriber("/camera_left/image_raw",Image,self.callback_left)
        self.image_right = rospy.Subscriber("/camera_right/image_raw",Image,self.callback_right)
        self.frame_recode = np.zeros([360,1280,3], np.uint8)
        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter("recode.mp4", self.codec, 60, (1280,360))

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

            self.frame_recode = self.main_frame.copy()

            self.blur = cv2.GaussianBlur(self.main_frame.copy(), (13, 13), 0)

            self.fgmask_1 = fgbg.apply(self.blur, None, 0.01)

            self.fgmask_erode = cv2.erode(self.fgmask_1, kernel_erosion_1, iterations = 1) #오픈 연산이아니라 침식으로 바꾸자

            self.fgmask_dila = cv2.dilate(self.fgmask_erode,kernel_dilation_2,iterations = 1)

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.fgmask_dila, connectivity = 8)


            for i in range(len(stats)):
                x, y, w, h, area = stats[i]

                aspect = w/h
                
                #if area > 3000 or area < 500 or aspect > 1.2 or aspect < 0.97 : 
                #    continue
                #print(aspect)
                cv2.rectangle(self.main_frame, (x, y), (x + w, y + h), (0,0,255), 3)
                cv2.putText(self.main_frame, str(aspect), (x - 1, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 



            t2 = time.time()
            cv2.imshow("main", self.main_frame)
            cv2.imshow("fgmask_1", self.fgmask_1)
            cv2.imshow("fgmask_erode", self.fgmask_erode)
            cv2.imshow("fgmask_dila", self.fgmask_dila)

            #print("fps : ",1/(t2-t1))
            #self.out.write(self.frame_recode)

            key = cv2.waitKey(10)

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