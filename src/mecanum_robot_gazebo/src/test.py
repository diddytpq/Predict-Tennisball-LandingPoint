import rospy
import numpy as np
from std_msgs.msg import Float64
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import sys, select, os
import roslib
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios

from tool.utils import *
import time



roslib.load_manifest('mecanum_robot_gazebo')




if __name__ == '__main__' :

    rospy.init_node('pingpong')


    mecanum_0 = Make_mecanum_left('mecanum_0')
    mecanum_1 = Make_mecanum_right('mecanum_1')

    mecanum_0.move(-10,0)

    """while True:
        mecanum_0.spwan_ball()
        mecanum_0.throw_ball()
        ball_landing_point = [mecanum_0.x_target, mecanum_0.y_target]
        
        mecanum_1.move(ball_landing_point[0],ball_landing_point[1])

        check_catch(mecanum_1)

        mecanum_1.spwan_ball()
        mecanum_1.throw_ball()  
        ball_landing_point = [mecanum_1.x_target, mecanum_1.y_target]

        mecanum_0.move(ball_landing_point[0],ball_landing_point[1])

        check_catch(mecanum_0)"""


    #mecanum_0.spwan_ball()
    #mecanum_1.spwan_ball()
    #mecanum_1.throw_ball()


