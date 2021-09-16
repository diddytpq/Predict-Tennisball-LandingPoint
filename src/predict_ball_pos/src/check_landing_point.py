import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import time

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)


roslib.load_manifest('mecanum_robot_gazebo')
ball_name = 'ball_left'


def gat_ball_stats():
    ball_state = g_get_state(model_name = "ball_left")

    return ball_state




if __name__ == "__main__" :
    
    while 1:
        ball_stats = gat_ball_stats()

        if 0 <ball_stats.pose.position.z < 0.005 :

            #print("real_landing_point = ",[ball_stats.pose.position.x, ball_stats.pose.position.y, ball_stats.pose.position.z], ",")
            print([np.round(ball_stats.pose.position.x,3), np.round(ball_stats.pose.position.y,3), np.round(ball_stats.pose.position.z,3)], ",")
