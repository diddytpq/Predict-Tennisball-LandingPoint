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


#rosservice call /gazebo/set_model_state '{model_state: { model_name: turtlebot3_waffle, pose: { position: { x: -1.55, y: 1.915 ,z: 0 }, orientation: {x: -1.72, y: 0.0015, z: 4.225, w: 0.999 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 7.66 } } , reference_frame: world } }'

roslib.load_manifest('ball_trajectory')


class Image_converter:

    def __init__(self):
        #self.image_pub = rospy.Publisher("/camera/rgb/image_raw",Image)

        self.bridge = CvBridge()
        rospy.init_node('Image_converter', anonymous=True)
        self.image_left = rospy.Subscriber("/camera_left/image_raw",Image,self.callback)
        #self.image_right = rospy.Subscriber("/camera_right/image_raw",Image,self.callback)


    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)

        (rows,cols,channels) = cv_image.shape
        if cols > 60 and rows > 60 :


            #print(cv_image.shape)

            self.main_img = cv2.resize(cv_image,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

                      

            cv2.imshow("main", self.main_img)


            key = cv2.waitKey(3)

            if key == 27 : 
                cv2.destroyAllWindows()
                
                return 0

        #try:
        #    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #except CvBridgeError as e:
        #    print(e)

def main(args):

    ic = Image_converter()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)