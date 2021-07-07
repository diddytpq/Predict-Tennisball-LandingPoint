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

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox



roslib.load_manifest('ball_trajectory')


conf_thres = 0.25
iou_thres=0.45
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
max_det = 1000 # maximum detections per image
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences
line_thickness=3,  # bounding box thickness (pixels)

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


conf_thres = 0.25
iou_thres=0.45
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
max_det = 1000 # maximum detections per image
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences
line_thickness=3,  # bounding box thickness (pixels)



class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        rospy.init_node('Image_converter', anonymous=True)

        self.image_left_0 = rospy.Subscriber("/camera_left_0/image_raw",Image,self.callback_left_0)
        self.image_left_1 = rospy.Subscriber("/camera_left_1/image_raw",Image,self.callback_left_1)
        self.image_right_0 = rospy.Subscriber("/camera_right_0/image_raw",Image,self.callback_right_0)
        self.image_right_1 = rospy.Subscriber("/camera_right_1/image_raw",Image,self.callback_right_1)

        self.frame_recode = np.zeros([360,1280,3], np.uint8)
        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.out_0 = cv2.VideoWriter("left.mp4", self.codec, 60, (640,640))
        self.out_1 = cv2.VideoWriter("right.mp4", self.codec, 60, (640,640))



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

            """self.left_data_0 = cv2.resize(self.left_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.left_data_1 = cv2.resize(self.left_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.right_data_0 = cv2.resize(self.right_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.right_data_1 = cv2.resize(self.right_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)"""

            self.left_frame = cv2.vconcat([self.left_data_0,self.right_data_0])
            self.right_frame = cv2.vconcat([self.left_data_1,self.right_data_1])

            #self.frame_recode = self.main_frame

            t2 = time.time()
            cv2.imshow("left_frame", self.left_frame)
            cv2.imshow("right_frame", self.right_frame)

            print(self.left_frame.shape)

            self.out_0.write(self.left_frame)
            self.out_1.write(self.right_frame)

            #print((t2-t1))
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