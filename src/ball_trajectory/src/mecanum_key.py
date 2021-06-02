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



roslib.load_manifest('ball_trajectory')

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


def check_velocity(cur_vel, max_vel):
    pass


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
    # start publisher of cmd_vel to control Turtlesim


    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum/wheel_1/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum/wheel_2/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum/wheel_3/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum/wheel_4/command", Float64, queue_size=10)
    
    linear = data[0]
    angular = data[1]

    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    robot_state = g_get_state(model_name="mecanum")

    print(linear[0], linear[1], linear[2])
    print(float(robot_state.twist.linear.x), float(robot_state.twist.linear.y), float(robot_state.twist.linear.z))


    x_vel = linear[0]
    y_vel = linear[1]
    z_vel = linear[2]

    z_angle = angular[2]

    vbx = 5.5 #km/h
    vby = 1.5 #km/h
    wbz = 3.5 #deg/sec

    twist = Twist()

    twist.linear.x = x_vel
    twist.linear.y = y_vel
    twist.linear.z = z_vel

    twist.angular.z = z_angle


    wheel_vel = mecanum_wheel_velocity(twist.linear.x, twist.linear.y, twist.angular.z)
    
    rospy.loginfo("\ttwist.linear.x : %f", twist.linear.x)
    rospy.loginfo("\ttwist.linear.y : %f", twist.linear.y)
    rospy.loginfo("\ttwist.linear.z : %f", twist.linear.z)
    
    # record values to log file and screen
    #rospy.loginfo("twist.linear.x: %f; twist.linear.y: %f ; angular %f", twist.linear.x, twist.linear.y, twist.angular.z)

    # publish cmd_vel move command to Turtlesim
    pub.publish(twist)
    pub_wheel_vel_1.publish(wheel_vel[0,:])
    pub_wheel_vel_2.publish(wheel_vel[1,:])
    pub_wheel_vel_3.publish(wheel_vel[2,:])
    pub_wheel_vel_4.publish(wheel_vel[3,:])


def stop_mecanum():
    # start publisher of cmd_vel to control Turtlesim


    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum/wheel_1/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum/wheel_2/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum/wheel_3/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum/wheel_4/command", Float64, queue_size=10)
    

    x_vel = 0
    y_vel = 0
    z_vel = 0

    vbx = 5.5 #km/h
    vby = 1.5 #km/h
    wbz = 3.5 #deg/sec

    twist = Twist()

    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0


    wheel_vel = mecanum_wheel_velocity(twist.linear.x, twist.linear.y, twist.angular.z)
    
    rospy.loginfo("\ttwist.linear.x : %f", twist.linear.x)
    rospy.loginfo("\ttwist.linear.y : %f", twist.linear.y)
    rospy.loginfo("\ttwist.linear.z : %f", twist.linear.z)
    
    # record values to log file and screen
    #rospy.loginfo("twist.linear.x: %f; twist.linear.y: %f ; angular %f", twist.linear.x, twist.linear.y, twist.angular.z)

    # publish cmd_vel move command to Turtlesim
    pub.publish(twist)
    pub_wheel_vel_1.publish(wheel_vel[0,:])
    pub_wheel_vel_2.publish(wheel_vel[1,:])
    pub_wheel_vel_3.publish(wheel_vel[2,:])
    pub_wheel_vel_4.publish(wheel_vel[3,:])


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
                move_mecanum([linear,angular])

            elif key == 'x' :
                linear[0] -= 1 
                move_mecanum([linear,angular])

            elif key == 'a' :
                linear[1] += 1 
                move_mecanum([linear,angular])

                move_mecanum([linear,angular])

            elif key == 'd' :
                linear[1] -= 1 
                move_mecanum([linear,angular])

                move_mecanum([linear,angular])

            elif key == 'q' :

                angular[2] -= 1 
                move_mecanum([linear,angular])

            elif key == 'e' :

                angular[2] += 1 
                move_mecanum([linear,angular])


            elif key == ' ' or key == 's' :
                linear = [0, 0, 0]
                angular = [0, 0, 0]
                move_mecanum([linear,angular])
            if (key == '\x03'):
                break

    except rospy.ROSInt:
        pass