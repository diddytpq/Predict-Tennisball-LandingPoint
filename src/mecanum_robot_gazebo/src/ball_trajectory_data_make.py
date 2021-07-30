import rospy
import sys, select, os
import roslib
import time

from tool.train_data_make_utils import *

roslib.load_manifest('mecanum_robot_gazebo')

if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_0 = Make_mecanum_left('mecanum_0')
    
    #mecanum_0.torque = [0, -20000, 0]
    
    #mecanum_0.torque = [0, 0, 0]
    #mecanum_0.torque = [0, 0, 20000]

    mecanum_0.del_ball()

    time.sleep(0.2)
    #mecanum_0.move(-11,0,mecanum_0,mecanum_1)

    add_catch_point = 3.5

    #f = open("ball_landing_data.txt",'w')


    while True:
        mecanum_0.spwan_ball("ball_left")
        mecanum_0.throw_ball()

        print([mecanum_0.x_target, mecanum_0.y_target])
        savedata = [mecanum_0.x_target, mecanum_0.y_target]
        #f.write(str(savedata) + ",")


        x_move = (np.random.randint(-13, -10))
        y_move = (np.random.randint(-4, 4))


        mecanum_0.move(x_move,y_move,mecanum_0)



        time.sleep(0.2)

