import rospy
import sys, select, os
import roslib
import time

from tool.tennis_test_utils import *

roslib.load_manifest('mecanum_robot_gazebo')

if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_L = Make_mecanum_left('mecanum_L')
    mecanum_R = Make_mecanum_right('mecanum_R')
    
    #mecanum_L.torque = [0, 0, 0]
    #mecanum_L.torque = [0, 109000, 0]
    #mecanum_L.torque = [0, -209000, 0]

    mecanum_R.torque = [0, -209000, 0]

    mecanum_R.ball_name = 'ball_right'
    mecanum_R.away_ball_name = "ball_left"
    mecanum_L.del_ball()
    mecanum_R.del_ball() 

    add_catch_point = 2.5
    add_catch_point = 3.5

    time.sleep(1)

    while True:
        mecanum_L.spwan_ball("ball_left")
        mecanum_L.throw_ball()
        
        #time.sleep(0.05)
        #ball_landing_point = [mecanum_0.x_target + add_catch_point * np.cos(mecanum_0.yaw_z), mecanum_0.y_target + add_catch_point * np.sin(mecanum_0.yaw_z)]
        #print(ball_landing_point)
        #print(add_catch_point * np.cos(mecanum_0.yaw_z),add_catch_point * np.sin(mecanum_0.yaw_z))


        mecanum_R.move(10,-10,mecanum_L)



        """mecanum_1.spwan_ball("ball_right")
        mecanum_1.throw_ball()  

        ball_landing_point = [mecanum_1.x_target - add_catch_point * np.cos(mecanum_1.yaw_z), mecanum_1.y_target - add_catch_point * np.sin(mecanum_1.yaw_z)]
        print(ball_landing_point)
        mecanum_0.move(ball_landing_point[0],ball_landing_point[1],mecanum_1)"""


        time.sleep(2)

        #mecanum_1.del_ball() 


