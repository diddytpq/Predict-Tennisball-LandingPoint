import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import time
from tool.mecanum_utils import *


roslib.load_manifest('mecanum_robot_gazebo')

g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

max_vel_forward = 1.5 # m/s
max_vel_lateral = 5.5 # m/s

ball_spawn_z = 1.5 # m
ball_init_vel_x = 40 #m/s
ball_init_vel_z = 3 #m/s

torque = [0, 209*1000, 0]

def get_position():

    robot_state = g_get_state(model_name="mecanum_R")

    object_pose = Pose()
    object_pose.position.x = float(robot_state.pose.position.x)
    object_pose.position.y = float(robot_state.pose.position.y)
    object_pose.position.z = float(robot_state.pose.position.z)

    object_pose.orientation.x = float(robot_state.pose.orientation.x)
    object_pose.orientation.y = float(robot_state.pose.orientation.y)
    object_pose.orientation.z = float(robot_state.pose.orientation.z)
    object_pose.orientation.w = float(robot_state.pose.orientation.w)
    
    roll_x, pitch_y, yaw_z = qua2eular(object_pose.orientation.x, object_pose.orientation.y,
                                        object_pose.orientation.z, object_pose.orientation.w)

    return object_pose.position.x, object_pose.position.y, object_pose.position.z

def vel_threshold(x_vel, y_vel):

    if x_vel > 0:
        if abs(x_vel) > max_vel_forward:
            x_vel = max_vel_forward

    elif x_vel < 0:
        if abs(x_vel) > max_vel_forward:
            x_vel = -max_vel_forward

    if y_vel > 0:
        if abs(y_vel) > max_vel_lateral:
            y_vel = max_vel_lateral

    elif y_vel < 0:
        if abs(y_vel) > max_vel_lateral:
            y_vel = -max_vel_lateral

    return x_vel, y_vel


def move_mecanum(linear,angular_z):

    pub = rospy.Publisher("/mecanum_R_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum_R/wheel_1_controller/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum_R/wheel_2_controller/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum_R/wheel_3_controller/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum_R/wheel_4_controller/command", Float64, queue_size=10)
    
    robot_state = g_get_state(model_name="mecanum_R")

    roll_x, pitch_y, yaw_z = qua2eular(robot_state.pose.orientation.x, 
                                        robot_state.pose.orientation.y, 
                                        robot_state.pose.orientation.z, 
                                        robot_state.pose.orientation.w)

    twist = Twist()

    x_vel, y_vel = vel_threshold(linear[0], linear[1])

    twist.linear.x = x_vel
    twist.linear.y = y_vel

    twist.angular.z = angular_z

    wheel_vel = mecanum_wheel_velocity(twist.linear.x, twist.linear.y, twist.angular.z)

    pub.publish(twist)
    pub_wheel_vel_1.publish(wheel_vel[0,:])
    pub_wheel_vel_2.publish(wheel_vel[1,:])
    pub_wheel_vel_3.publish(wheel_vel[2,:])
    pub_wheel_vel_4.publish(wheel_vel[3,:])

    return [x_vel, y_vel], angular_z


def spwan_ball():

    del_ball()
    time.sleep(0.5)

    file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/tennis_ball/ball_main.sdf'
    srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    robot_x, robot_y, robot_z = get_position()

    ball_pose = Pose()
    ball_pose.position.x = robot_x
    ball_pose.position.y = robot_y
    ball_pose.position.z = robot_z + ball_spawn_z


    file_xml = open(file_localition)
    xml_string=file_xml.read()

    req = SpawnModelRequest()
    req.model_name = "ball_right"
    req.model_xml = xml_string
    req.initial_pose = ball_pose

    res = srv_spawn_model(req)

def del_ball():
    srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    res = srv_delete_model("ball_right")

def ball_apply_force(target, force, torque, duration):
    
    rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=10)

    apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

    wrench = Wrench()
    wrench.force = Vector3(*force)
    wrench.torque = Vector3(*torque)
    success = apply_wrench(
        target + '::ball_link',
        'world',
        Point(0, 0, 0),
        wrench,
        rospy.Time().now(),
        rospy.Duration(duration))

def throw_ball():

    spwan_ball()

    duration = 0.001

    ror_matrix = rotation_matrix(np.deg2rad(0))

    v0 = np.sqrt(ball_init_vel_x**2 + ball_init_vel_z**2)
    launch_angle = np.arctan(ball_init_vel_z/v0)

    force = [-v0 * 0.057 / duration, 0, ball_init_vel_z * 0.057 / duration]
    
    apply_force, apply_torque = get_wrench(force, torque, ror_matrix)

    ball_apply_force("ball_right", apply_force, apply_torque, duration)

    t0 = time.time()

    print("vx, vz :", ball_init_vel_x, ball_init_vel_z)


def gat_ball_stats():
    ball_state = g_get_state(model_name = "ball_right")

    return ball_state


