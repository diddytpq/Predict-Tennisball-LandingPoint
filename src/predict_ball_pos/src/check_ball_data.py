import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_srvs.srv import Empty
from std_msgs.msg import String, Float64, Float64MultiArray
from nav_msgs.msg import Odometry

import cv2

import pickle

import time

roslib.load_manifest('mecanum_robot_gazebo')

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
# reset_sim_time = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

ball_name = 'ball_left'
#ball_name = 'ball_right'

pre_z = np.nan
pre_gradient = np.nan
esti_ball_landing_point = []

def callback_landing_point(data):
    global esti_ball_landing_point

    if len(data.data) < 1 :
        return 0

    #esti_ball_landing_point = [data.data[0], data.data[1], data.data[2]]

    esti_ball_landing_point = list(data.data)

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

def landing_point(ball_stats, real_ball_landing_point_list, esti_ball_landing_point_list):
    if check_bounce(ball_stats.pose.position.z) :
        #real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)])
        real_ball_landing_point_list.append([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3)])
        esti_ball_landing_point_list.append(esti_ball_landing_point)
        
        print("real_ball_landing_point_list = " ,real_ball_landing_point_list[-1], len(real_ball_landing_point_list))
        print("esti_ball_landing_point_list = " ,esti_ball_landing_point_list[-1], len(esti_ball_landing_point_list))

    return real_ball_landing_point_list, esti_ball_landing_point_list



def real_ball_trajectory(ball_stats, real_trajectory, time):

    if np.round(ball_stats.pose.position.x,3) == 0 or np.round(ball_stats.pose.position.x,3) > 13.:
        return real_trajectory

    real_trajectory.append([now.nsecs,np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)])

    return real_trajectory



def check_plus(num):

    if num > 0:
        return True
    else:
        return False

if __name__ == "__main__" :

    esti_ball_landing_point_list = []
    real_ball_landing_point_list = []

    real_trajectory = []
    real_data = []


    rospy.init_node('check_landing_point', anonymous=True)

    img = np.zeros((10,10,3), np.uint8)

    while True:
        t0 = time.time()
        #rospy.Subscriber("esti_landing_point", Float64MultiArray, callback_landing_point)
        ball_stats = gat_ball_stats()
        now = rospy.get_rostime()


        # record ball landing point
        #real_ball_landing_point_list, esti_ball_landing_point_list = landing_point(ball_stats, real_ball_landing_point_list, esti_ball_landing_point_list)

        # record ball trajectory

        real_trajectory = real_ball_trajectory(ball_stats, real_trajectory, now)

        if len(real_trajectory) > 2:

            if check_plus(real_trajectory[-2][1]) == True and check_plus(real_trajectory[-1][1]) == False:
                print(np.array(real_trajectory[:-2]).shape)
                real_data.append(real_trajectory[:-2])
                real_trajectory = []
                # reset_sim_time()


        print(len(real_data))
        print(time.time() - t0)

        cv2.imshow("empty Windows",img)

        key = cv2.waitKey(1)

        if key == 27 or len(real_ball_landing_point_list) == 100 or len(real_data) == 100:

            # with open('real_ball_list.bin', 'wb') as f:
            #     pickle.dump(np.array(real_ball_landing_point_list),f)

            # with open('esti_ball_list.bin', 'wb') as f:
            #     pickle.dump(np.array(esti_ball_landing_point_list),f)
            
            with open('data/real_.bin', 'wb') as f:
                pickle.dump(real_data,f)
        
            break