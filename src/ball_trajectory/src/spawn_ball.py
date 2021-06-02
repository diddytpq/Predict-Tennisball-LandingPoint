import sys
import roslib
roslib.load_manifest('ball_trajectory')
import rospy
import os
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
from std_msgs.msg import Empty as EmptyMsg
from nav_msgs.msg import Odometry

#https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/get_robot_position.py

def throw_ball():
    pass



def spwan_ball(model_name):

    srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    req = DeleteModelRequest()
    req.model_name = model_name    

    res = srv_delete_model(model_name)

    file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball.urdf'
    srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)



    file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball_main.sdf'
    srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    
    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    
    state = g_get_state(model_name="turtlebot3_waffle")

    object_pose = Pose()
    object_pose.position.x = float(state.pose.position.x)
    object_pose.position.y = float(state.pose.position.y)
    object_pose.position.z = float(state.pose.position.z + 1.2)

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

def main(args):

    rospy.init_node("object_spawner", anonymous=True)

    model_name = "ball"
    spwan_ball(model_name)

    #print(xml_string)

    #input()

    throw_ball()

if __name__ == '__main__':
    main(sys.argv)