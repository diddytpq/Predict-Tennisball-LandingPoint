import rospy
import sys, select, os
import roslib
import time

from tool.pingpong_utils_ver2 import *

roslib.load_manifest('mecanum_robot_gazebo')

if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_L = Make_mecanum_left('mecanum_L')
    mecanum_R = Make_mecanum_right('mecanum_R')
    

    mecanum_R.torque = [0, -209000, 0]

    mecanum_R.ball_name = 'ball_right'
    mecanum_R.away_ball_name = "ball_left"
    mecanum_L.del_ball()
    mecanum_R.del_ball() 

    add_catch_point = 0


    while True:
        mecanum_L.spwan_ball("ball_left")
        mecanum_L.throw_ball()
        #time.sleep(0.05)
        ball_landing_point = [mecanum_L.x_target + add_catch_point * np.cos(mecanum_L.yaw_z), mecanum_L.y_target + add_catch_point * np.sin(mecanum_L.yaw_z)]
        #print(ball_landing_point)
        #print(add_catch_point * np.cos(mecanum_0.yaw_z),add_catch_point * np.sin(mecanum_0.yaw_z))


        mecanum_R.move(ball_landing_point[0],ball_landing_point[1],mecanum_L)


        mecanum_R.spwan_ball("ball_right")
        mecanum_R.throw_ball()  
        #time.sleep(0.05)
        ball_landing_point = [mecanum_R.x_target - add_catch_point * np.cos(mecanum_R.yaw_z), mecanum_R.y_target - add_catch_point * np.sin(mecanum_R.yaw_z)]
        mecanum_L.move(ball_landing_point[0],ball_landing_point[1],mecanum_R)


