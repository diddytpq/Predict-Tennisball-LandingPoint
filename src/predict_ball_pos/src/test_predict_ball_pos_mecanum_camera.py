from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.append(path + '/lib')


import numpy as np
import ray

import rospy
import cv2

import time
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torch
import argparse
from heatmap_based_object_tracking.models.network import *
#from heatmap_based_object_tracking.models.network_b0_ver2 import *

from heatmap_based_object_tracking.utils import *

from tools import *
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
                    default='/lib/heatmap_based_object_tracking/weights/gazebo.tar', help='input model weight for predict')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--record', type=bool, default=False,
                    help='record option')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ', torch.cuda.is_available())

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)


model = EfficientNet(1.2, 1.4) # b3 width_coef = 1.2, depth_coef = 1.4

model.to(device)

checkpoint = torch.load(path + '/' + args.load_weight)
model.load_state_dict(checkpoint['state_dict'])

model.eval()


@ray.remote
class Img_Buffer(object):

    def __init__(self):
        rospy.init_node('Img_Buffer', anonymous=True)

        rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image, self.callback_camera)
        rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image, self.callback_depth)

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
            #print("depth",time.time())

        except CvBridgeError as e:
            print(e)

    def get_img(self):

        return self.img_data, self.depth_data

def get_ball_status():

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

    return ball_pose.position.x, ball_pose.position.y, ball_pose.position.z

def get_robot_pos():

    robot_state = g_get_state(model_name='mecanum_R')

    object_pose = Pose()
    object_pose.position.x = float(robot_state.pose.position.x)
    object_pose.position.y = float(robot_state.pose.position.y)
    object_pose.position.z = float(robot_state.pose.position.z)

    object_pose.orientation.x = float(robot_state.pose.orientation.x)
    object_pose.orientation.y = float(robot_state.pose.orientation.y)
    object_pose.orientation.z = float(robot_state.pose.orientation.z)
    object_pose.orientation.w = float(robot_state.pose.orientation.w)
    
    # angle = qua2eular(object_pose.orientation.x, object_pose.orientation.y,
    #                     object_pose.orientation.z, object_pose.orientation.w)

    return object_pose.position.x , object_pose.position.y, object_pose.position.z 

def main():

    rospy.init_node('predict_ball_pos_node', anonymous=True)

    input_img_buffer = []

    ball_trajectory = []
    real_ball_trajectory = []

    esti_ball_trajectory_data = []
    
    init_robot_pos_x = 12.88
    init_robot_pos_y = 0
    init_robot_pos_z = 1.0

    ball_disappear_cnt = 0

    real_data = []
    esti_data = []

    img_buffer = Img_Buffer.remote()

    BTE = Ball_Trajectory_Estimation()

    while True:
        camera_data = ray.get(img_buffer.get_img.remote())

        if len(camera_data[0]):
            print('---------------------------------------------------')

            #ball_x, ball_y, ball_z = get_ball_status()
            now = rospy.get_rostime()

            t1 = time.time()
            frame = camera_data[0]
            depth = camera_data[1]
            robot_x, robot_y, robot_z =  get_robot_pos()
            
            img = cv2.resize(frame,(WIDTH, HEIGHT))

            input_img_buffer.append(img)

            if len(input_img_buffer) < 3:
                continue

            if len(input_img_buffer) > 3:
                input_img_buffer = input_img_buffer[-3:]


            unit = tran_input_img(input_img_buffer)
            unit = torch.from_numpy(unit).to(device, dtype=torch.float)

            torch.cuda.synchronize()

            with torch.no_grad():

                unit = unit / 255

                h_pred = model(unit)
                h_pred = (h_pred * 255).cpu().numpy()
                
                h_pred = (h_pred[0]).astype('uint8')
                h_pred = np.asarray(h_pred).transpose(1, 2, 0)

                h_pred = (170 < h_pred) * h_pred

                torch.cuda.synchronize()

            frame, depth_list, ball_cand_pos, ball_cand_score = find_ball_center_base(h_pred, frame, np.float32(depth), ratio_w, ratio_h)
            # frame, depth_list, ball_cand_pos, ball_cand_score = find_ball_v3(h_pred, frame, np.float32(depth), ratio_w, ratio_h)
            
            ball_pos = cal_ball_pos(ball_cand_pos, depth_list)

            if len(ball_pos):
                if ball_pos[0] < 13:
                    x_pos_dt =  robot_x - init_robot_pos_x
                    y_pos_dt =  robot_y - init_robot_pos_y

                    ball_trajectory.append([now.nsecs, init_robot_pos_x - (ball_pos[0] - x_pos_dt), init_robot_pos_y + (ball_pos[1] + y_pos_dt), ball_pos[2] + init_robot_pos_z])
                    
                    #ball_trajectory.append([ball_pos[0], ball_pos[1], ball_pos[2]])

                    #array2data.data = ball_pos
                    #pub.publish(array2data)

                    if len(ball_trajectory) > 2:
                        print("ball_trajectory",len(ball_trajectory))

                        measure_time = np.array(ball_trajectory.copy())[:,0]
                        measure_data = np.array(ball_trajectory.copy())[:,1:]

                        esti_ball_trajectory = BTE.cal_rebound_trajectory(measure_data, dt = time.time() - t1)

                        if esti_ball_trajectory:
                            # print((measure_data))
                            # print(len(esti_ball_trajectory))
                            # real_ball_trajectory.append([ball_x, ball_y, ball_z])
                            
                            esti_ball_trajectory_data.append([measure_time, measure_data, esti_ball_trajectory])

                else:
                    ball_disappear_cnt += 1
            
            else:
                ball_disappear_cnt += 1

            if ball_disappear_cnt > 30:
                ball_disappear_cnt = 0

                if len(esti_ball_trajectory_data) > 8:
                #     #real_data.append([real_ball_trajectory])
                #     #print(len(real_data))

                #     # real_data.append(real_ball_trajectory)
                    esti_data.append(esti_ball_trajectory_data)
                
                ball_trajectory = []
                esti_ball_trajectory_data = []
                BTE.clear()
                print(len(esti_data))


            #print("FPS : ",1 / (time.time() - t1))

            cv2.imshow("img",frame)
            #cv2.imshow("h_pred",h_pred)

            key = cv2.waitKey(1)
            if key == 27 or len(esti_data) == 100 :

                cv2.destroyAllWindows()
                ray.shutdown()

                with open('data/esti_.bin', 'wb') as f:
                    pickle.dump((esti_data),f)
                break


if __name__ == '__main__':

    main() 