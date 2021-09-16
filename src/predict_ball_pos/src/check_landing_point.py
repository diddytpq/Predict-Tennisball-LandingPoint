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


pre_z = np.nan
pre_gradient = np.nan

esti_ball_landing_point_list = []

def callback_landing_point(data):
    global esti_ball_landing_point_list

    esti_ball_landing_point_list.append(data)



def gat_ball_stats():
    ball_state = g_get_state(model_name = "ball_left")

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



if __name__ == "__main__" :
    
    real_ball_landing_point_list = []

    while True:
        ball_stats = gat_ball_stats()
        rospy.Subscriber("/esti_landing_point", Float64MultiArray, callback_landing_point)

        if check_bounce(ball_stats.pose.position.z) :
            real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)])

            print(len(real_ball_landing_point_list))
            print(len(esti_ball_landing_point_list))


        key = cv2.watikey()

        if key == ord('s'):

            with open('real_ball_list.bin', 'wb') as f:
                pickle.dump(np.array(real_ball_landing_point_list),f)

            with open('esti_ball_list.bin', 'wb') as f:
                pickle.dump(np.array(esti_ball_landing_point_list),f)
            

            break