import rospy
import sys, select, os
import roslib
import time
import pickle

from tool.tennis_test_utils import *

roslib.load_manifest('mecanum_robot_gazebo')


"""def callback_landing_point(data):
    global esti_ball_landing_point

    if len(data.data) < 1 :
        return 0

    #esti_ball_landing_point = [data.data[0], data.data[1], data.data[2]]

    esti_ball_landing_point = list(data.data)
"""



if __name__ == '__main__' :

    rospy.init_node('pingpong')

    mod = 1

    mecanum_L = Make_mecanum_left('mecanum_L')
    mecanum_R = Make_mecanum_right('mecanum_R')
    
    mecanum_L.torque = [0, 209000, 0]
    mecanum_R.torque = [0, -209000, 0]

    mecanum_R.ball_name = 'ball_right'
    mecanum_R.away_ball_name = "ball_left"
    mecanum_L.del_ball()
    mecanum_R.del_ball()

    add_catch_point = 1

    while True:
        mecanum_L.spwan_ball("ball_left")
        mecanum_L.throw_ball()


        ball_landing_point = [mecanum_L.x_target + add_catch_point * np.cos(mecanum_L.yaw_z), mecanum_L.y_target + add_catch_point * np.sin(mecanum_L.yaw_z)]


        print("ball_landing_point",ball_landing_point)

        ball_landing_point = [mecanum_L.x_target + add_catch_point * np.cos(mecanum_L.yaw_z), mecanum_L.y_target + (add_catch_point * np.sin(mecanum_L.yaw_z))]

        mecanum_R.move_based_mecanum_camera(ball_landing_point[0],ball_landing_point[1] ,mecanum_L)
        # mecanum_R.move(ball_landing_point[0],ball_landing_point[1] ,mecanum_L)

        t0 = time.time()
        while True:
            # return_home(mecanum_R)
            if (time.time() - t0) > 3: break

        #check = input()

        # if check == 'c':
        # print(len(mecanum_R.esti_data))
        # if len(mecanum_R.esti_data) > 15:
        #     #mecanum_R.save_data()
        #     break


