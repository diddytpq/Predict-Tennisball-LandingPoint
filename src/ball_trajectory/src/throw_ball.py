import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from nav_msgs.msg import Odometry
import time
from tool.utils import  *


roslib.load_manifest('ball_trajectory')



def spwan_ball(model_name):

    srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    req = DeleteModelRequest()
    req.model_name = model_name    

    res = srv_delete_model(model_name)

    time.sleep(0.2)

    file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball_main.sdf'
    srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
  
    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    
    state = g_get_state(model_name="mecanum")

    object_pose = Pose()
    object_pose.position.x = float(state.pose.position.x)
    object_pose.position.y = float(state.pose.position.y)
    object_pose.position.z = float(state.pose.position.z + 1)

    object_pose.orientation.x = float(state.pose.orientation.x)
    object_pose.orientation.y = float(state.pose.orientation.y)
    object_pose.orientation.z = float(state.pose.orientation.z)
    object_pose.orientation.w = float(state.pose.orientation.w)


    file_xml = open(file_localition)
    xml_string=file_xml.read()

    req = SpawnModelRequest()
    
    req.model_name = model_name
    req.model_xml = xml_string
    req.initial_pose = object_pose

    res = srv_spawn_model(req)



def throw_ball():


    starting_time = 0
    duration = 0.01
    force = [91.96, 0, 34.06]
    torque = [0, 0, 0]

    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    
    state = g_get_state(model_name="mecanum")


    x = float(state.pose.orientation.x)
    y = float(state.pose.orientation.y)
    z = float(state.pose.orientation.z)
    w = float(state.pose.orientation.w)

    roll_x, pitch_y, yaw_z  = qua2eular(x,y,z,w)

    
    rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=10)

    apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
    body_name = 'ball::base_link'

    wrench = Wrench()
    ror_matrix = rotation_matrix(yaw_z)
    force, torque = get_wrench(force, torque, ror_matrix)

    wrench.force = Vector3(*force)
    wrench.torque = Vector3(*torque)
    success = apply_wrench(
        body_name,
        'world',
        Point(0, 0, 0),
        wrench,
        rospy.Time().now(),
        rospy.Duration(duration))

    if success:
        print('Body wrench perturbation applied!')
        #print('\tFrame: ', body_name)
        print('\tDuration [s]: ', duration)
        print('\tForce [N]: ', force)
        print('\tTorque [Nm]: ', torque)
        print('\tlaunch angle: ',np.rad2deg(np.arctan(force[2]/force[0])))

        v0, rpm = cal(force, torque)

        print('\tv0: {} \t RPM: {}' .format(v0, rpm))

    else:
        print('Failed!')


if __name__ == '__main__':

    rospy.init_node('set_body_wrench')

    model_name = "ball"
    spwan_ball(model_name)   
    time.sleep(0.5)
    throw_ball()
