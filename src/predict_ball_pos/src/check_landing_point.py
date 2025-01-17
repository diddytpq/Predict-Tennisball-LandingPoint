import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import String, Float64, Float64MultiArray
from nav_msgs.msg import Odometry

import cv2

import pickle

import time

roslib.load_manifest('mecanum_robot_gazebo')

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)



ball_name = 'ball_left'
#ball_name = 'ball_right'


pre_z = np.nan
pre_gradient = np.nan
esti_ball_landing_point = []

def callback_landing_point(data):
    global esti_ball_landing_point

    if len(data.data) < 1 or np.isnan(data.data[0]):
        return 0

    #esti_ball_landing_point = [data.data[0], data.data[1], data.data[2]]

    esti_ball_landing_point.append([data.data[0], data.data[1], data.data[2]])

    



def gat_ball_stats():
    ball_state = g_get_state(model_name = ball_name)

    return ball_state


def check_bounce(cnt_z):
    global pre_z
    global pre_gradient

    if np.isnan(pre_z):
        
        pre_z = cnt_z

        return False

    cnt_gradient = cnt_z - pre_z

    if np.isnan(pre_gradient):

        pre_gradient = cnt_gradient

        return False

    if check_grad(cnt_gradient) == True and check_grad(pre_gradient) == False:
        
        pre_gradient = cnt_gradient
        
        return True
    else:
        pre_gradient = cnt_gradient

        return False

def check_grad(num):

    if num < 0 :
        return False

    else :
        return True


def empty(x):
    pass

if __name__ == "__main__" :
    try:
        esti_ball_landing_point_list = []
        real_ball_landing_point_list = []

        rospy.init_node('check_landing_point', anonymous=True)

        img = np.zeros((10,10,3), np.uint8)

        # rospy.Subscriber("esti_landing_point", Float64MultiArray, callback_landing_point)
        rospy.Subscriber("/esti_landing_point_robot_camera", Float64MultiArray, callback_landing_point)

        while True:
            ball_stats = gat_ball_stats()

            if check_bounce(ball_stats.pose.position.z) and len(esti_ball_landing_point):
                #real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)])
                real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3)])
                # esti_ball_landing_point_list.append(np.mean(np.array(esti_ball_landing_point[-3]), axis = 0)[:2])
                esti_ball_landing_point_list.append(np.array(esti_ball_landing_point[-2])[:2])
                print(esti_ball_landing_point)
                print(len(esti_ball_landing_point))
                print(np.mean(np.array(esti_ball_landing_point)))

                esti_ball_landing_point = []
                
                print("real_ball_landing_point_list = " ,real_ball_landing_point_list[-1], len(real_ball_landing_point_list))
                print("esti_ball_landing_point_list = " ,esti_ball_landing_point_list[-1], len(esti_ball_landing_point_list))

            cv2.imshow("empty Windows",img)

            key = cv2.waitKey(1)

            if key == ord('s') or len(real_ball_landing_point_list) == 100 or (len(real_ball_landing_point_list) != len(esti_ball_landing_point_list)):

                with open('real_ball_list1.bin', 'wb') as f:
                    pickle.dump(np.array(real_ball_landing_point_list),f)

                with open('esti_ball_list1.bin', 'wb') as f:
                    pickle.dump(np.array(esti_ball_landing_point_list),f)
                

                break
    except:

        with open('real_ball_list1.bin', 'wb') as f:
                pickle.dump(np.array(real_ball_landing_point_list),f)

        with open('esti_ball_list1.bin', 'wb') as f:
            pickle.dump(np.array(esti_ball_landing_point_list),f)