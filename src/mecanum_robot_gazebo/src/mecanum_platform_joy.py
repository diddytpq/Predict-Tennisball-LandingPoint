#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

# called when joy message is received
def mecanum_wheel_velocity(vx, vy, wz):
    r = 0.0762 # radius of wheel
    l = 0.23 #length between {b} and wheel
    w = 0.25225 #depth between {b} abd wheel
    alpha = l + w
    
    q_dot = np.array([wz, vx, vy])
    J_pseudo = np.array([[-alpha, 1, -1],[alpha, 1, 1],[alpha, 1, -1],[alpha, 1,1]])

    print(np.reshape(q_dot,(3,1)))

    u = 1/r * J_pseudo @ np.reshape(q_dot,(3,1))#q_dot.T

    return u

def tj_callback(data):
    # start publisher of cmd_vel to control Turtlesim
    
    pub = rospy.Publisher("/mecanum_L_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum_L/wheel_1_controller/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum_L/wheel_2_controller/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum_L/wheel_3_controller/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum_L/wheel_4_controller/command", Float64, queue_size=10)
    
    
    vbx = 5.5 #km/h
    vby = 1.5 #km/h
    wbz = 3.5 #deg/sec

    twist = Twist()

    twist.linear.x = vbx*data.axes[1]
    twist.linear.y = vby*data.axes[0]
    twist.angular.z = wbz*data.axes[2]

    wheel_vel = mecanum_wheel_velocity(twist.linear.x, twist.linear.y, twist.angular.z)
    
    rospy.loginfo("axes[0] data : %f", data.axes[0])
    rospy.loginfo("axes[1] data : %f", data.axes[1])
    rospy.loginfo("axes[2] data : %f", data.axes[2])
    
    # record values to log file and screen
    rospy.loginfo("twist.linear.x: %f; twist.linear.y: %f ; angular %f", twist.linear.x, twist.linear.y, twist.angular.z)

    # publish cmd_vel move command to Turtlesim
    pub.publish(twist)
    pub_wheel_vel_1.publish(wheel_vel[0,:])
    pub_wheel_vel_2.publish(wheel_vel[1,:])
    pub_wheel_vel_3.publish(wheel_vel[2,:])
    pub_wheel_vel_4.publish(wheel_vel[3,:])

def joy_listener():
    # start node
    rospy.init_node("mecanum_platform_joy", anonymous=True)

    # subscribe to joystick messages on topic "joy"
    rospy.Subscriber("joy", Joy, tj_callback, queue_size=1)

    # keep node alive until stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        joy_listener()
    except rospy.ROSInt:
        pass
