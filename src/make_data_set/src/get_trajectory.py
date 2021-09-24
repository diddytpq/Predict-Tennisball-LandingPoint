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
esti_ball_landing_point = []

def callback_landing_point(data):
    global esti_ball_landing_point

    if len(data.data) < 1 :
        return 0

    esti_ball_landing_point = [data.data[0], data.data[1], data.data[2]]


    



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


def empty(x):
    pass

if __name__ == "__main__" :

    esti_ball_landing_point_list = []
    real_ball_landing_point_list = []

    srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    res = srv_delete_model("ball_left")

    rospy.init_node('check_landing_point', anonymous=True)

    img = np.zeros((10,10,3), np.uint8)


    while True:
        ball_stats = gat_ball_stats()


        if ball_stats.pose.position.z > 0.00001 :
            real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)])
                
            print("real_ball_landing_point_list = " ,real_ball_landing_point_list[-1], len(real_ball_landing_point_list))

        cv2.imshow("empty Windows",img)

        key = cv2.waitKey(1)

        if key == ord('s') or ball_stats.pose.position.x > 15 :

            with open('real_ball_traj.bin', 'wb') as f:
                pickle.dump(np.array(real_ball_landing_point_list),f)



            break