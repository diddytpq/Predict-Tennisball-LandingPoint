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

from pathlib import Path
import torch
import argparse
from heatmap_based_object_tracking.network import *
from heatmap_based_object_tracking.utils import *

BATCH_SIZE = 1
HEIGHT=288
WIDTH=512

height = 720
width = 1280

ratio_h = height / HEIGHT
ratio_w = width / WIDTH
size = (width, height)

parser = argparse.ArgumentParser(description='Gazebo simualation')

parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='heatmap_based_object_tracking/weights/gazebo.tar', help='input model weight for predict')
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

model = efficientnet_b3()

model.to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
checkpoint = torch.load(path + '/' + args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']

input_img_buffer = []

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image,self.callback_depth)
        rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image,self.callback_camera)

        #rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.main)

        if args.record == True:
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            self.out_0 = cv2.VideoWriter("record_video.mp4", self.codec, 30, (1280,720))

    def callback_camera(self, data):
        try:
            rospy.loginfo(1)

            self.t0 = time.time()
            self.camera_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)
     
    def callback_depth(self, data):
        try:
            rospy.loginfo(2)

            self.camera_depth_data = self.bridge.imgmsg_to_cv2(data, "passthrough")


        except CvBridgeError as e:
            print(e)


    def main(self, data):
        
        rospy.loginfo(3)
        global input_img_buffer

        (rows,cols,channels) = self.camera_data.shape

        if cols > 60 and rows > 60 :


            self.camera_depth_color = np.uint8(self.camera_depth_data) 
            self.camera_depth_color = cv2.applyColorMap(self.camera_depth_color * 20, cv2.COLORMAP_JET)
            #self.main_frame = cv2.hconcat([self.camera_data, self.camera_depth_color])

            frame = self.camera_data
            depth_img = self.camera_depth_data

            img = cv2.resize(frame,(WIDTH, HEIGHT))

            input_img_buffer.append(img)

            if len(input_img_buffer) < 3:
                return 0

            if len(input_img_buffer) > 3:
                input_img_buffer = input_img_buffer[-3:]

            t1 = time.time()

            unit = tran_input_img(input_img_buffer)

            unit = torch.from_numpy(unit).to(device, dtype=torch.float)
            torch.cuda.synchronize()

            with torch.no_grad():

                h_pred = model(unit)

                torch.cuda.synchronize()
                t2 = time.time()

                """h_pred = (h_pred * 255).cpu().numpy()
                
                h_pred = (h_pred[0]).astype('uint8')
                h_pred = np.asarray(h_pred).transpose(1, 2, 0)

                h_pred = (100 < h_pred) * h_pred

            frame, depth_list, ball_cand_pos, ball_cand_score = find_ball_v3(h_pred, frame, depth_img, ratio_w, ratio_h)"""

            # cv2.imshow("image",frame)
            # cv2.imshow("h_pred",h_pred)

            # print((t2 - t1))
            # print("FPS : ",1 / (t2 - t1))

            #print((t2 - self.t0))
            #print("FPS : ",1 / (t2 - self.t0))

            if args.record == True:

                self.out_0.write(self.main_frame)

            key = cv2.waitKey(1)

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
