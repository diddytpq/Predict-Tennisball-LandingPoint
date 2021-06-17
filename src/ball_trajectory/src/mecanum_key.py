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

from throw_ball import *


roslib.load_manifest('ball_trajectory')

def qua2eular(x,y,z,w):

    q_x = x
    q_y = y
    q_z = z
    q_w = w

    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def getKey():
    if os.name == 'nt':
      if sys.version_info[0] >= 3:
        return msvcrt.getch().decode()
      else:
        return msvcrt.getch()

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def check_velocity(cur_vel):
    max_x = 5.5 #km/h
    max_y = 3.3 #km/h
    max_wz = 3.5 #deg/sec

    x_vel, y_vel, z_vel, z_angle = cur_vel

    if max_x < abs(x_vel):
        if x_vel > 0: x_vel = max_x
        else: x_vel = -max_x

    if max_y < abs(y_vel):
        if y_vel > 0: y_vel = max_y
        else: y_vel = -max_y

    if max_wz < abs(z_angle):
        if z_angle > 0: z_angle = max_wz
        else: z_angle = -max_wz
        

    return [x_vel, y_vel, z_vel], z_angle

def mecanum_wheel_velocity(vx, vy, wz):
    r = 0.0762 # radius of wheel
    l = 0.23 #length between {b} and wheel
    w = 0.25225 #depth between {b} abd wheel
    alpha = l + w
    
    q_dot = np.array([wz, vx, vy])
    J_pseudo = np.array([[-alpha, 1, -1],[alpha, 1, 1],[alpha, 1, -1],[alpha, 1,1]])

    u = 1/r * J_pseudo @ np.reshape(q_dot,(3,1))#q_dot.T

    return u




def move_mecanum(data):
    # start publisher of cmd_vel to control mecanum


    """pub = rospy.Publisher("/mecanum_0_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum/wheel_1/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum/wheel_2/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum/wheel_3/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum/wheel_4/command", Float64, queue_size=10)"""
    
    pub = rospy.Publisher("/mecanum_1_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum_1/wheel_1/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum_1/wheel_2/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum_1/wheel_3/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum_1/wheel_4/command", Float64, queue_size=10)
    
    linear = data[0]
    angular = data[1]

    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    robot_state = g_get_state(model_name="mecanum_1")


    linear, angular[2] = check_velocity([linear[0],linear[1],linear[2],angular[2]])
    
    roll_x, pitch_y, yaw_z = qua2eular(robot_state.pose.orientation.x, robot_state.pose.orientation.y, 
                            robot_state.pose.orientation.z, robot_state.pose.orientation.w)

    print(np.rad2deg(yaw_z))
    twist = Twist()

    twist.linear.x = linear[0]
    twist.linear.y = linear[1]
    twist.linear.z = linear[2]

    twist.angular.z = angular[2]


    wheel_vel = mecanum_wheel_velocity(twist.linear.x, twist.linear.y, twist.angular.z)
    
    print("-------------------------------------------------------------------------------")
    rospy.loginfo("\ttwist.linear.x : %f", twist.linear.x)
    rospy.loginfo("\ttwist.linear.y : %f", twist.linear.y)
    rospy.loginfo("\ttwist.linear.z : %f", twist.angular.z)
    
    # record values to log file and screen
    #rospy.loginfo("twist.linear.x: %f; twist.linear.y: %f ; angular %f", twist.linear.x, twist.linear.y, twist.angular.z)

    # publish cmd_vel move command to mecanum 
    pub.publish(twist)
    pub_wheel_vel_1.publish(wheel_vel[0,:])
    pub_wheel_vel_2.publish(wheel_vel[1,:])
    pub_wheel_vel_3.publish(wheel_vel[2,:])
    pub_wheel_vel_4.publish(wheel_vel[3,:])

    return [linear[0],linear[1],linear[2]], angular[2]


if __name__ == '__main__':
    try:
        rospy.init_node('mecanum_key')
        if os.name != 'nt':
            settings = termios.tcgetattr(sys.stdin)
        linear = [0, 0, 0]
        angular = [0, 0, 0]
        while(1):

            key = getKey()

            if key == 'w' :

                linear[0] += 1 
                linear, angular[2] = move_mecanum([linear,angular])

            elif key == 'x' :
                linear[0] -= 1 
                linear, angular[2] = move_mecanum([linear,angular])


            elif key == 'a' :
                linear[1] += 1 
                linear, angular[2] = move_mecanum([linear,angular])


            elif key == 'd' :
                linear[1] -= 1 
                linear, angular[2] = move_mecanum([linear,angular])

            elif key == 'q' :

                angular[2] += 1 
                linear, angular[2] = move_mecanum([linear,angular])


            elif key == 'e' :

                angular[2] -= 1 
                linear, angular[2] = move_mecanum([linear,angular])


            elif key == 's' :
                linear = [0, 0, 0]
                angular = [0, 0, 0]
                linear, angular[2] = move_mecanum([linear,angular])
                

            elif key == ' ' :
                linear = [0, 0, 0]
                angular = [0, 0, 0]
                linear, angular[2] = move_mecanum([linear,angular])

                model_name = "ball"
                spwan_ball(model_name)   
                time.sleep(0.1)
                throw_ball()

            if (key == '\x03'):
                linear = [0, 0, 0]
                angular = [0, 0, 0]
                linear, angular[2] = move_mecanum([linear,angular])
                break

    except rospy.ROSInt:
        pass