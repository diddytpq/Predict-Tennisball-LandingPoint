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

class Make_mecanum_left():

    def __init__(self, model_name):
        self.model_name = model_name
        
        self.pub = rospy.Publisher("/{}_vel".format(model_name), Twist, queue_size=10)
        self.pub_wheel_vel_1 = rospy.Publisher("/{}/wheel_1/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_2 = rospy.Publisher("/{}/wheel_2/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_3 = rospy.Publisher("/{}/wheel_3/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_4 = rospy.Publisher("/{}/wheel_4/command".format(model_name), Float64, queue_size=10)

        self.g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.vel_forward = 5.5 #m/s
        self.vel_lateral = 1.5 #m/s
        self.ball_fly_time = 0.45 #max height time [sec]
        self.vel_forward_apply = 0
        self.vel_lateral_apply = 0
        self.amax = 3

        self.spawn_pos_z = 0.5


        self.ball_name = 'ball_left'
        
        self.torque = [0,200000,0]
        self.delete_model_name = "ball_left"

        self.twist = Twist()
        self.get_position()

    def get_position(self):

        self.robot_state = self.g_get_state(model_name=self.model_name)

        self.object_pose = Pose()
        self.object_pose.position.x = float(self.robot_state.pose.position.x)
        self.object_pose.position.y = float(self.robot_state.pose.position.y)
        self.object_pose.position.z = float(self.robot_state.pose.position.z)

        self.object_pose.orientation.x = float(self.robot_state.pose.orientation.x)
        self.object_pose.orientation.y = float(self.robot_state.pose.orientation.y)
        self.object_pose.orientation.z = float(self.robot_state.pose.orientation.z)
        self.object_pose.orientation.w = float(self.robot_state.pose.orientation.w)
       
        self.angle = qua2eular(self.object_pose.orientation.x, self.object_pose.orientation.y,
                            self.object_pose.orientation.z, self.object_pose.orientation.w)

        #print(self.object_pose.position.x, self.object_pose.position.y, self.object_pose.position.z)
        #print(self.angle)
   
    def check_velocity(self, x_vel, y_vel):

        if self.vel_forward < abs(x_vel):
            if x_vel > 0: x_vel = self.vel_forward
            else: x_vel = -self.vel_forward

        if self.vel_lateral < abs(y_vel):
            if y_vel > 0: y_vel = self.vel_lateral
            else: y_vel = -self.vel_lateral
            
        return x_vel, y_vel

    def set_x_velocity(self,dt):

        if self.x_error > 0:
            self.vel_forward_apply += self.amax * dt
            if abs(self.vel_forward_apply) > self.vel_forward:
                self.vel_forward_apply = self.vel_forward

        else :
            self.vel_forward_apply -= self.amax * dt
            if abs(self.vel_forward_apply) > self.vel_forward:
                self.vel_forward_apply = -self.vel_forward


    def set_y_velocity(self,dt):

        if self.y_error > 0:
            self.vel_lateral_apply += self.amax * dt
            if abs(self.vel_lateral_apply) > self.vel_lateral:
                self.vel_lateral_apply = self.vel_lateral

        else :
            self.vel_lateral_apply -= self.amax * dt
            if abs(self.vel_lateral_apply) > self.vel_lateral:
                self.vel_lateral_apply = -self.vel_lateral   

    def stop(self):
        self.vel_forward_apply = 0
        self.vel_lateral_apply = 0
        self.twist = Twist()

        self.twist.linear.x = self.vel_forward_apply
        self.twist.linear.y = self.vel_lateral_apply
        self.twist.linear.z = 0
        self.twist.angular.z = 0 

        self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)
        self.pub.publish(self.twist)
        self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
        self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
        self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
        self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])


    def move(self, x_target, y_target, my_mecanum):
        t0 = time.time()
        dt = 0
        while True:
            if ball_catch_check(my_mecanum, "ball_left"):
                self.stop()
                break 

            self.get_position()
            t1 = time.time()

            dt = t1 - t0


            self.x_error = x_target - self.object_pose.position.x
            self.y_error = y_target - self.object_pose.position.y

            if (abs(self.x_error) <0.1 and abs(self.y_error)< 0.1) :
                self.stop()
                
            else :
                self.set_x_velocity(dt)
                self.set_y_velocity(dt)

                if abs(self.x_error) < 0.1:
                    self.vel_forward_apply = 0

                if abs(self.y_error) < 0.1:
                    self.vel_lateral_apply = 0

                self.twist.linear.x = self.vel_forward_apply
                self.twist.linear.y = self.vel_lateral_apply
                self.twist.linear.z = 0

                self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

                self.pub.publish(self.twist)
                self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
                self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
                self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
                self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])
            t0 = time.time()
            
    def spwan_ball(self, name):
        self.cnt = 0

        #time.sleep(0.1)
        #print("________________________________________________")
        file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball_main.sdf'
        #file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/soccer_ball.sdf'
        
        srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
        self.get_position()
        ball_pose = Pose()
        ball_pose.position.x = self.object_pose.position.x
        ball_pose.position.y = self.object_pose.position.y
        ball_pose.position.z = self.object_pose.position.z + self.spawn_pos_z

        self.ball_pre_position_z = self.object_pose.position.z + self.spawn_pos_z
        self.pre_gradient = 1

        ball_pose.orientation.x = 0
        ball_pose.orientation.y = 0
        ball_pose.orientation.z = 0
        ball_pose.orientation.w = 1


        file_xml = open(file_localition)
        xml_string=file_xml.read()

        req = SpawnModelRequest()
        req.model_name = name
        req.model_xml = xml_string
        req.initial_pose = ball_pose

        res = srv_spawn_model(req)

    def set_ball_target(self):
        self.x_target = (np.random.randint(6, 10) + np.random.rand())
        self.y_target = (np.random.randint(-3, 3) + np.random.rand())

        #self.x_target = 20
        #self.y_target = 0

        self.get_position()
        
        self.x_error = self.x_target - self.object_pose.position.x
        self.y_error = self.y_target - self.object_pose.position.y
        self.s = np.sqrt(self.x_error**2 + self.y_error**2)


    def throw_ball(self):

        duration = 0.001

        self.set_ball_target()

        self.yaw_z = np.arctan(self.y_error/self.x_error)
        self.ror_matrix = rotation_matrix(self.yaw_z)
        vz0 = 9.8 * self.ball_fly_time

        h = (self.object_pose.position.z + self.spawn_pos_z) + vz0 * self.ball_fly_time - (9.8 * self.ball_fly_time**2)/2
        self.ball_fly_time_plus = np.sqrt(2 * h / 9.8)
        v0 = self.s/(self.ball_fly_time + self.ball_fly_time_plus)

        self.v = np.sqrt(v0**2 + vz0**2)
        self.launch_angle = np.arctan(vz0/v0)

        self.force = [v0 * 0.057 * 1000, 0, vz0 * 0.057 *1000 ]

        self.apply_force, self.apply_torque = get_wrench(self.force, self.torque, self.ror_matrix)

        self.ball_apply_force(self.apply_force, self.apply_torque, duration)

        #self.apply_torque = [int(self.apply_torque[0]),int(self.apply_torque[1]),int(self.apply_torque[2])]
        #print(v0)

    def ball_apply_force(self, force, torque, duration):

        rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=10)

        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        wrench = Wrench()
        wrench.force = Vector3(*force)
        wrench.torque = Vector3(*torque)
        success = apply_wrench(
            self.ball_name + '::ball_link',
            'world',
            Point(0, 0, 0),
            wrench,
            rospy.Time().now(),
            rospy.Duration(duration))

        self.get_ball_stats()

        self.ball_pre_vel_linear_x = self.ball_vel.linear.x  
        self.ball_pre_vel_linear_y = self.ball_vel.linear.y  
        self.ball_pre_vel_linear_z = self.ball_vel.linear.z  

        self.ball_pre_vel_angular_x = self.ball_vel.angular.x  
        self.ball_pre_vel_angular_y = self.ball_vel.angular.y  
        self.ball_pre_vel_angular_z = self.ball_vel.angular.z  


    def del_ball(self):
        srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        
        #req = DeleteModelRequest()
        #req.model_name = "ball_right" 


        res = srv_delete_model(self.delete_model_name)

        #time.sleep(0.1)

    def get_ball_stats(self):
        self.ball_state = self.g_get_state(model_name = self.ball_name)

        self.ball_pose = Pose()
        self.ball_pose.position.x = float(self.ball_state.pose.position.x)
        self.ball_pose.position.y = float(self.ball_state.pose.position.y)
        self.ball_pose.position.z = float(self.ball_state.pose.position.z)
        
        self.ball_vel = Twist()

        self.ball_vel.linear.x = float(self.ball_state.twist.linear.x)
        self.ball_vel.linear.y = float(self.ball_state.twist.linear.y)
        self.ball_vel.linear.z = float(self.ball_state.twist.linear.z)

        self.ball_vel.angular.x = float(self.ball_state.twist.angular.x)
        self.ball_vel.angular.y = float(self.ball_state.twist.angular.y)
        self.ball_vel.angular.z = float(self.ball_state.twist.angular.z)

    def break_ball_rolling(self):

        self.get_ball_stats()
        duration = 0.001


        if self.check_bounce() and self.ball_pose.position.z < 0.021 :
            self.cnt += 1
            #print(self.cnt)

            self.get_ball_stats()

            #print(self.ball_vel.linear.x , self.ball_pre_vel_linear_x)

            w_y2 = self.ball_vel.angular.y - 1.5 * 0.033 * (self.ball_vel.linear.x - self.ball_pre_vel_linear_x) / 0.03 ** 2
            w_x2 = self.ball_vel.angular.x - 1.5 * 0.033 * (self.ball_vel.linear.y - self.ball_pre_vel_linear_y) / 0.03 ** 2
            
            force = [0, 0, 0]

            #print("w_y2 : ",w_y2)
            self.apply_torque = [-(self.ball_vel.angular.x  - w_x2) * 1000, -(self.ball_vel.angular.y - w_y2) * 1000, 0]
            
            #print(self.apply_torque[0]/100 , self.apply_torque[1]/1000)

            self.ball_apply_force(force, self.apply_torque, duration)
            
    def check_bounce(self):

        self.current_gradient = self.ball_pose.position.z - self.ball_pre_position_z

        if self.check_gradient(self.pre_gradient) == False and self.check_gradient(self.current_gradient) == True:
            self.ball_pre_position_z = self.ball_pose.position.z
            self.pre_gradient = self.current_gradient
            return True

        else:
            self.ball_pre_position_z = self.ball_pose.position.z
            self.pre_gradient = self.current_gradient
            return False
        
    def check_gradient(self, gradient):

        if gradient < 0: 
            return False
        
        else: 
            return True


def ball_catch_check(mecanum, ball_name):



    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    ball_state = g_get_state(model_name = ball_name)

    mecanum.get_position()

    ball_x = ball_state.pose.position.x

    if abs(ball_x) > 18:
        mecanum.del_ball()
        return True

    return False


