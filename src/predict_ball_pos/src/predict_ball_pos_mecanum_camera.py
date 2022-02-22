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

from multiprocessing import Process, Pipe, Manager
from multiprocessing.managers import BaseManager

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



camera_data = []
camera_depth_data = []


#rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.main)

class image_pipe(object):
    
    def __init__(self):
        self.img_data = []
        self.depth_data = []

    def img_upload(self, data_list):
        self.img_data = data_list

    def depth_upload(self, data_list):
        self.depth_data = data_list

    def download(self):
        return self.img_data, self.depth_data

def get_gazebo_img(img_pipe):

    class Img_Buffer():

        def __init__(self):
            rospy.init_node('Img_Buffer', anonymous=True)

            self.img_data = []
            self.depth_data = []
            self.bridge = CvBridge()


        def callback_camera(self, data):
            try:
                self.img_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

            except CvBridgeError as e:
                print(e)


        def callback_depth(self, data):
            try:

                self.depth_data = self.bridge.imgmsg_to_cv2(data, "passthrough")

            except CvBridgeError as e:
                print(e)

    img_buffer = Img_Buffer()

    #rospy.init_node('Image_converter', anonymous=True)

    time.sleep(1)
    print('----------------serial start--------------------')
    rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image, img_buffer.callback_camera)
    rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image, img_buffer.callback_depth)


    try:
        while True:
            if len(img_buffer.img_data):
                img_pipe.img_upload(img_buffer.img_data)
                img_pipe.depth_upload(img_buffer.depth_data)

            else:
                print('not img')
                time.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down")
        


def main(args):

    BaseManager.register('image_pipe', image_pipe)
    manager = BaseManager()
    manager.start()
    inst = manager.image_pipe()

    process = Process(target=get_gazebo_img, args=[inst])
    process.start()

    input_img_buffer = []

    while True:

        camera_data = inst.download()
        
        if len(camera_data[0]):

            print("--------------------------------------")
            
            frame = camera_data[0]
            depth = camera_data[1]

            img = cv2.resize(frame,(WIDTH, HEIGHT))

            input_img_buffer.append(img)

            if len(input_img_buffer) < 3:
                continue

            if len(input_img_buffer) > 3:
                input_img_buffer = input_img_buffer[-3:]

            t1 = time.time()

            unit = tran_input_tensor(input_img_buffer, device)

            torch.cuda.synchronize()

            with torch.no_grad():

                h_pred = model(unit)
                h_pred = (h_pred * 255).cpu().numpy()
                
                h_pred = (h_pred[0]).astype('uint8')
                h_pred = np.asarray(h_pred).transpose(1, 2, 0)

                h_pred = (100 < h_pred) * h_pred

                torch.cuda.synchronize()

            frame, depth_list, ball_cand_pos, ball_cand_score = find_ball_v3(h_pred, frame, depth, ratio_w, ratio_h)
            
            print("ball_cand_pos : ",ball_cand_pos)
            print("depth_list : ",depth_list)

            depth_img = cv2.applyColorMap(np.uint8(depth) * 20, cv2.COLORMAP_JET)

            main_frame = cv2.resize(cv2.hconcat([frame, depth_img]), dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)


            cv2.imshow("main_frame",main_frame)
            cv2.imshow("h_pred",h_pred)

            t2 = time.time()

            print((t2 - t1))
            print("FPS : ",1 / (t2 - t1))

            key = cv2.waitKey(1)

            if key == 27 : 
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main(sys.argv)
