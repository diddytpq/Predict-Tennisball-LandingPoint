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
ball_init_vel_x = 20 #m/s
ball_init_vel_z = 5 #m/s

torque = [0, 209*1000, 0]

def get_position():

    robot_state = g_get_state(model_name="mecanum_L")

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

    pub = rospy.Publisher("/mecanum_L_vel", Twist, queue_size=10)
    pub_wheel_vel_1 = rospy.Publisher("/mecanum_L/wheel_1_controller/command", Float64, queue_size=10)
    pub_wheel_vel_2 = rospy.Publisher("/mecanum_L/wheel_2_controller/command", Float64, queue_size=10)
    pub_wheel_vel_3 = rospy.Publisher("/mecanum_L/wheel_3_controller/command", Float64, queue_size=10)
    pub_wheel_vel_4 = rospy.Publisher("/mecanum_L/wheel_4_controller/command", Float64, queue_size=10)
    
    robot_state = g_get_state(model_name="mecanum_L")

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

    file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/tennis_ball/ball_test.sdf'
    srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    robot_x, robot_y, robot_z = get_position()

    ball_pose = Pose()
    ball_pose.position.x = robot_x
    ball_pose.position.y = robot_y
    ball_pose.position.z = robot_z + ball_spawn_z


    file_xml = open(file_localition)
    xml_string=file_xml.read()

    req = SpawnModelRequest()
    req.model_name = "ball_left"
    req.model_xml = xml_string
    req.initial_pose = ball_pose

    res = srv_spawn_model(req)

def del_ball():
    srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    res = srv_delete_model("ball_left")

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

    force = [v0 * 0.057 / duration, 0, ball_init_vel_z * 0.057 / duration]
    
    apply_force, apply_torque = get_wrench(force, torque, ror_matrix)

    ball_apply_force("ball_left", apply_force, apply_torque, duration)

    t0 = time.time()

    print("vx, vz :", ball_init_vel_x, ball_init_vel_z)

    while gat_ball_stats().pose.position.z > 0.02 :
        t1 = time.time()
        dt = t1 - t0
        cal_liftdrag(dt)

        t0 = time.time()


def gat_ball_stats():
    ball_state = g_get_state(model_name = "ball_left")

    return ball_state

def cal_liftdrag(dt):

    dt = 0.05

    ball_state = gat_ball_stats()

    down_motion = 0

    ball_vel_xy = np.sqrt((ball_state.twist.linear.x ** 2) + (ball_state.twist.linear.y ** 2))
    ball_vel_xyz =  np.sqrt((ball_state.twist.linear.x ** 2) + (ball_state.twist.linear.y ** 2 + (ball_state.twist.linear.z ** 2)))

    if ball_state.twist.angular.y > 0:
        ball_angular_xy = np.sqrt((ball_state.twist.angular.x ** 2) + (ball_state.twist.angular.y ** 2))

    else:
        ball_angular_xy = -np.sqrt((ball_state.twist.angular.x ** 2) + (ball_state.twist.angular.y ** 2))

    angle_x = np.arctan(ball_state.twist.linear.y / ball_state.twist.linear.x)
    angle_xy = np.arctan(ball_state.twist.linear.z / ball_vel_xy)

    cd = 0.507
    cl = -0.645 * 0.033 * ball_angular_xy / ball_vel_xy

    drag_force = -0.5 * cd * 1.2041 * np.pi * (0.033 ** 2) * ball_vel_xyz
    lift_force = 0.5 * cl * 1.2041 * np.pi * (0.033 ** 2) * ball_vel_xyz

    if ball_state.twist.linear.z < 0:
        down_motion = 1

    if down_motion == 0 : 

        if cl < 0:

            drag_force_z = drag_force * np.sin(angle_xy)
            drag_force_xy = drag_force * np.cos(angle_xy)
            drag_force_x = drag_force_xy * np.cos(angle_x)
            drag_force_y = drag_force_xy * np.sin(angle_x)
            
            lift_force_z = lift_force * np.sin(angle_xy)
            lift_force_xy = lift_force * np.cos(angle_xy)
            lift_force_x = -lift_force_xy * np.cos(angle_x)
            lift_force_y = lift_force_xy * np.sin(angle_x)
                
        else:

            drag_force_z = drag_force * np.sin(angle_xy)
            drag_force_xy = drag_force * np.cos(angle_xy)
            drag_force_x = drag_force_xy * np.cos(angle_x)
            drag_force_y = drag_force_xy * np.sin(angle_x)
            
            lift_force_z = lift_force * np.sin(angle_xy)
            lift_force_xy = lift_force * np.cos(angle_xy)
            lift_force_x = -lift_force_xy * np.cos(angle_x)
            lift_force_y = lift_force_xy * np.sin(angle_x)
        
    else:

        if cl < 0:

            drag_force_z = drag_force * np.sin(angle_xy)
            drag_force_xy = drag_force * np.cos(angle_xy)
            drag_force_x = drag_force_xy * np.cos(angle_x)
            drag_force_y = drag_force_xy * np.sin(angle_x)
            
            lift_force_z = - lift_force * np.sin(angle_xy)
            lift_force_xy = lift_force * np.cos(angle_xy)
            lift_force_x = lift_force_xy * np.cos(angle_x)
            lift_force_y = lift_force_xy * np.sin(angle_x)

        else:

            drag_force_z = drag_force * np.sin(angle_xy)
            drag_force_xy = drag_force * np.cos(angle_xy)
            drag_force_x = drag_force_xy * np.cos(angle_x)
            drag_force_y = drag_force_xy * np.sin(angle_x)
            
            lift_force_z = -lift_force * np.sin(angle_xy)
            lift_force_xy = lift_force * np.cos(angle_xy)
            lift_force_x = lift_force_xy * np.cos(angle_x)
            lift_force_y = -lift_force_xy * np.sin(angle_x)


    liftdrag_force_x = drag_force_x + lift_force_x
    liftdrag_force_y = drag_force_y + lift_force_y
    liftdrag_force_z = drag_force_z + lift_force_z
    
    """print("----------------------------------")
    print("ball postion : {}, {}, {}".format(np.round(ball_state.pose.position.x,3), np.round(ball_state.pose.position.y,3), np.round(ball_state.pose.position.z,3)))
    print("ball_velocity : {}, {}, {}".format(np.round(ball_state.twist.linear.x,3), np.round(ball_state.twist.linear.y,3), np.round(ball_state.twist.linear.z,3)))
    print("drag force : {}, {}, {}".format(np.round(drag_force_x,3), np.round(drag_force_y,3), np.round(drag_force_z,3)))
    print("lift force : {}, {}, {}".format(np.round(lift_force_x,3), np.round(lift_force_y,3), np.round(lift_force_z,3)))
    print("liftdrag force : {}, {}, {}".format(np.round(liftdrag_force_x,5), np.round(liftdrag_force_y,5), np.round(liftdrag_force_z,5)))"""

    force = [np.round(liftdrag_force_x,5) / dt, np.round(liftdrag_force_y,5) / dt, np.round(liftdrag_force_z,5) / dt]
    
    ball_apply_force("ball_left", force, [0,0,0], dt)
