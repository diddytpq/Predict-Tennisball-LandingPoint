import rospy
import sys, select, os
import roslib
import time
import argparse

from tool.tennis_match_utils import *


parser = argparse.ArgumentParser(description = 'tennis_match')

parser.add_argument('--mod', type = int, default='0', help = 'set tennis match mod')
args = parser.parse_args()



if __name__ == '__main__' :

    rospy.init_node('pingpong')

    mecanum_L = Make_mecanum_left('mecanum_L')
    mecanum_R = Make_mecanum_right('mecanum_R')
    
    mecanum_L.torque = [0, 209000, 0]
    mecanum_R.torque = [0, -209000, 0]

    mecanum_R.ball_name = 'ball_right'
    mecanum_R.away_ball_name = "ball_left"
    mecanum_L.del_ball()
    mecanum_R.del_ball() 

    add_catch_point = 1

    if args.mod == 0: # check mecanum move and ball launch

        while True:

            add_catch_point = 2
            
            mecanum_L.spwan_ball("ball_left")
            mecanum_L.throw_ball()
            
            ball_landing_point = [mecanum_L.x_target + add_catch_point * np.cos(mecanum_L.yaw_z), mecanum_L.y_target + add_catch_point * np.sin(mecanum_L.yaw_z)]
            #print(ball_landing_point)
            #print(add_catch_point * np.cos(mecanum_0.yaw_z),add_catch_point * np.sin(mecanum_0.yaw_z))
            

            mecanum_R.move(ball_landing_point[0],ball_landing_point[1] ,mecanum_L)


            mecanum_R.spwan_ball("ball_right")
            mecanum_R.throw_ball()  
            ball_landing_point = [mecanum_R.x_target - add_catch_point * np.cos(mecanum_R.yaw_z), mecanum_R.y_target - add_catch_point * np.sin(mecanum_R.yaw_z)]
            mecanum_L.move(ball_landing_point[0],ball_landing_point[1] ,mecanum_R)



    if args.mod == 1: #predict ball pos by camera

        while True:
            mecanum_L.spwan_ball("ball_left")
            mecanum_L.throw_ball()
            

            mecanum_R.move_base_camera(add_catch_point, mecanum_L)


            mecanum_R.spwan_ball("ball_right")
            mecanum_R.throw_ball()  

            ball_landing_point = [mecanum_R.x_target - add_catch_point * np.cos(mecanum_R.yaw_z), mecanum_R.y_target - add_catch_point * np.sin(mecanum_R.yaw_z)]
        
            mecanum_L.move(ball_landing_point[0],ball_landing_point[1] ,mecanum_R)

