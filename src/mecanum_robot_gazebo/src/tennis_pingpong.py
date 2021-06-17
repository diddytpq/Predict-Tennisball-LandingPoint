import rospy
import sys, select, os
import roslib
import time

from tool.pingpong_utils import *

roslib.load_manifest('mecanum_robot_gazebo')

if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_0 = Make_mecanum_left('mecanum_0')
    mecanum_1 = Make_mecanum_right('mecanum_1')
    
    mecanum_0.del_ball()
    mecanum_1.del_ball()

    left_score = 0
    right_score = 0

    add_catch_point = 1.5

    score_board(left_score, right_score, "_")

    while True:
        mecanum_0.spwan_ball("ball_left")
        mecanum_0.throw_ball()
        ball_landing_point = [mecanum_0.x_target + add_catch_point * np.cos(mecanum_0.yaw_z), mecanum_0.y_target + add_catch_point * np.sin(mecanum_0.yaw_z)]
        
        time.sleep(0.05)

        mecanum_1.move(ball_landing_point[0],ball_landing_point[1],mecanum_0)

        left_score, right_score = ball_catch_check(mecanum_1, "ball_left", left_score, right_score)
        mecanum_1.spwan_ball("ball_right")
        mecanum_1.throw_ball()  
        ball_landing_point = [mecanum_1.x_target - add_catch_point * np.cos(mecanum_1.yaw_z), mecanum_1.y_target - add_catch_point * np.sin(mecanum_1.yaw_z)]

        time.sleep(0.05)

        mecanum_0.move(ball_landing_point[0],ball_landing_point[1],mecanum_1)

        left_score, right_score = ball_catch_check(mecanum_0,"ball_right", left_score, right_score)

