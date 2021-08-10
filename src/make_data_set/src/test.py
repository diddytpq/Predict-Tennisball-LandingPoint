import rospy
import sys, select, os
import roslib
import time

from tool.test_utils import *

roslib.load_manifest('mecanum_robot_gazebo')

if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_0 = Make_mecanum_left('mecanum_0')
    
    #mecanum_0.torque = [0, -2000000, 0]
    #mecanum_0.torque = [0, 0, 2000000]
    mecanum_0.torque = [0, 0, 0]
    

    mecanum_0.del_ball()

    time.sleep(0.2)
    #mecanum_0.move(-12.2,1,mecanum_0)

    while True:
        mecanum_0.spwan_ball("ball_left")
        mecanum_0.throw_ball()

        mecanum_0.move(-12.2,1,mecanum_0)



        time.sleep(1)

