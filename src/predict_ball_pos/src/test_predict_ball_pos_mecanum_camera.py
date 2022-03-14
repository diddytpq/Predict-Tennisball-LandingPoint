#!/usr/bin/env python

from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.append("/heatmap_based_object_tracking")

import numpy as np
import time
import roslib
import rospy
import cv2
from std_msgs.msg import String, Float64, Float64MultiArray
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torch
import argparse
from heatmap_based_object_tracking.models.network import *
from heatmap_based_object_tracking.utils import *

from multiprocessing import Process, Pipe, Manager
from multiprocessing.managers import BaseManager

import pickle

BATCH_SIZE = 1
HEIGHT=288
WIDTH=512

height = 360
width = 640

ratio_h = height / HEIGHT
ratio_w = width / WIDTH
size = (width, height)

parser = argparse.ArgumentParser(description='Gazebo simualation')

parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight', type=str,
                    default='/heatmap_based_object_tracking/weights/gazebo.tar', help='input model weight for predict')
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())

model = EfficientNet(1.2, 1.4) # b3 width_coef = 1.2, depth_coef = 1.4

model.to(device)
if args.optimizer == 'Ada':
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
checkpoint = torch.load(path + '/' + args.load_weight)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

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
                #rospy.loginfo(time.time())

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
        
def get_ball_status():

    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    ball_state = g_get_state(model_name = 'ball_left')

    ball_pose = Pose()
    ball_pose.position.x = float(ball_state.pose.position.x)
    ball_pose.position.y = float(ball_state.pose.position.y)
    ball_pose.position.z = float(ball_state.pose.position.z)
    
    ball_vel = Twist()

    ball_vel.linear.x = float(ball_state.twist.linear.x)
    ball_vel.linear.y = float(ball_state.twist.linear.y)
    ball_vel.linear.z = float(ball_state.twist.linear.z)

    ball_vel.angular.x = float(ball_state.twist.angular.x)
    ball_vel.angular.y = float(ball_state.twist.angular.y)
    ball_vel.angular.z = float(ball_state.twist.angular.z)

    return ball_pose.position.x, ball_pose.position.y, ball_pose.position.z ,ball_vel.linear.x, ball_vel.linear.y, ball_vel.linear.z

def main(args):
    #global camera_data, camera_depth_data
    BaseManager.register('image_pipe', image_pipe)
    manager = BaseManager()
    manager.start()
    inst = manager.image_pipe()

    process = Process(target=get_gazebo_img, args=[inst])
    process.start()

    #a = rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image,callback_depth)
    #b = rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image,callback_camera)
    rospy.init_node('Predict_ball_pos', anonymous=True)

    pub = rospy.Publisher('/esti_ball_pos',Float64MultiArray, queue_size = 1)
    array2data = Float64MultiArray()

    input_img_buffer = []

    robot_pos_x = 11.5
    robot_pos_y = 0
    robot_pos_z = 1.0

    ball_trajectory = []
    real_ball_trajectory = []

    ball_disappear_cnt = 0

    real_data = []
    while True:

        camera_data = inst.download()
        
        ball_x, ball_y, ball_z, ball_vel_x, ball_vel_y, ball_vel_z = get_ball_status()
        

        if len(camera_data[0]):
            
            frame = camera_data[0]
            depth = camera_data[1]

            img = cv2.resize(frame,(WIDTH, HEIGHT))

            input_img_buffer.append(img)

            if len(input_img_buffer) < 3:
                continue

            if len(input_img_buffer) > 3:
                input_img_buffer = input_img_buffer[-3:]

            t1 = time.time()

            # unit = tran_input_img(input_img_buffer)
            # unit = torch.from_numpy(unit).to(device, dtype=torch.float)

            unit = tran_input_tensor(input_img_buffer, device)

            torch.cuda.synchronize()

            with torch.no_grad():

                unit = unit / 255

                h_pred = model(unit)
                h_pred = (h_pred * 255).cpu().numpy()
                
                h_pred = (h_pred[0]).astype('uint8')
                h_pred = np.asarray(h_pred).transpose(1, 2, 0)

                h_pred = (200 < h_pred) * h_pred

                torch.cuda.synchronize()

            frame, depth_list, ball_cand_pos, ball_cand_score = find_ball_v3(h_pred, frame, np.float32(depth), ratio_w, ratio_h)

            #depth_list = [11.5 - ball_x]

            ball_pos = cal_ball_pos(ball_cand_pos, depth_list)
            #print("depth_list", depth_list)

            if len(ball_pos):
                if ball_pos[0] < 15:
                    print('---------------------------------------------------')
                    #print("ball_pos",[ball_pos[0], ball_pos[1], ball_pos[2]])

                    #print("real_ball_pos",[ball_x, ball_y, ball_z])

                    #ball_trajectory.append([robot_pos_x - ball_pos[0], ball_pos[1] - robot_pos_y, robot_pos_z + ball_pos[2]])
                    real_ball_trajectory.append([ball_x, ball_y, ball_z])
                    
                    ball_trajectory.append([ball_pos[0], ball_pos[1], ball_pos[2]])

                    array2data.data = ball_pos
                    pub.publish(array2data)



                else:
                    ball_disappear_cnt += 1
            
            else:
                ball_disappear_cnt += 1

            if ball_disappear_cnt > 15:
                ball_disappear_cnt = 0
                print("real_ball_trajectory = ",real_ball_trajectory)

                if len(real_ball_trajectory):
                    real_data.append([real_ball_trajectory])
                    print(len(real_data))
                
                ball_trajectory = []
                real_ball_trajectory = []
            
            """if ball_disappear_cnt > 15:

                ball_trajectory = []
                ball_disappear_cnt = 0

            if len(ball_trajectory) > 1:

                vel_x, vel_y = ball_vel_check(ball_trajectory)

                print("vel_x, vel_y",vel_x, vel_y)
                print("real_vel_x, real_vel_y",ball_vel_x, ball_vel_y)
            """

            #depth = depth * 2.55

            frame = cv2.resize(frame,(1280, 720))

            cv2.imshow("image",frame)
            #cv2.imshow("depth",np.uint8(depth))

            #cv2.imshow("h_pred",h_pred)

            t2 = time.time()

            #print((t2 - t1))
            #print("FPS : ",1 / (t2 - t1))

            key = cv2.waitKey(1)

            if key == ord('c'):
                #print("ball_trajectory = ",ball_trajectory)
                print("real_ball_trajectory = ",real_ball_trajectory)

                ball_trajectory = []
                real_ball_trajectory = []


            if key == 27 : 
                cv2.destroyAllWindows()
                with open('real_ball_list.bin', 'wb') as f:
                    pickle.dump((real_data),f)
                break

if __name__ == '__main__':
    main(sys.argv)
