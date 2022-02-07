import rospy
import sys
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Float64, Float64MultiArray
from nav_msgs.msg import Odometry
import time
from tool.mecanum_utils import *

roslib.load_manifest('mecanum_robot_gazebo')

class Make_mecanum_left():

    def __init__(self, model_name):
        self.model_name = model_name
        
        self.pub = rospy.Publisher("/{}_vel".format(model_name), Twist, queue_size=10)
        self.pub_wheel_vel_1 = rospy.Publisher("/{}/wheel_1_controller/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_2 = rospy.Publisher("/{}/wheel_2_controller/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_3 = rospy.Publisher("/{}/wheel_3_controller/command".format(model_name), Float64, queue_size=10)
        self.pub_wheel_vel_4 = rospy.Publisher("/{}/wheel_4_controller/command".format(model_name), Float64, queue_size=10)

        self.g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.vel_forward = 1.5 #m/s
        self.vel_lateral = 5.5 #m/s
        
        self.ball_fly_time = 0.45 #max height time [sec]
        self.vel_forward_apply = 0
        self.vel_lateral_apply = 0
        self.amax = 3

        self.spawn_pos_z = 1.3

        self.ball_name = 'ball_left'
        self.away_ball_name = 'ball_right'
        self.away_ball_vel_max_x = 0
        self.away_ball_vel_max_y = 0
        
        self.duration = 0.01
        self.torque = [0,209000,0]
        self.delete_model_name = "ball_right"

        self.twist = Twist()
        self.get_position()
        self.score = 0

        self.ball_preposition_list_z = [self.spawn_pos_z]
        self.pre_gradient_z = [1]

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


    def move(self, x_target, y_target, away_mecanum):
        t0 = time.time()

        while True:
            return_home(away_mecanum)

            if self.ball_catch_check():

                self.stop()
                away_mecanum.stop()
                break 

            self.get_position()

            t1 = time.time()
            self.dt = t1 - t0

            #self.cal_liftdrag()
            
            t0 = time.time()

            #print("dt : ", self.dt)

            
            self.x_error = x_target - self.object_pose.position.x
            self.y_error = y_target - self.object_pose.position.y
            
            #print(self.x_error, self.y_error)
            if (abs(self.x_error) <0.1 and abs(self.y_error)< 0.1) :
                self.stop()
                away_mecanum.stop()
        
            else:
                self.set_x_velocity(self.dt)
                self.set_y_velocity(self.dt)
                if abs(self.x_error) < 0.1:
                    self.vel_forward_apply = 0

                if abs(self.y_error) < 0.1:
                    self.vel_lateral_apply = 0


                self.twist = Twist()
                #print(self.vel_forward_apply, self.vel_lateral_apply)
                
                self.twist.linear.x = self.vel_forward_apply
                self.twist.linear.y = self.vel_lateral_apply
                self.twist.linear.z = 0

                self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

                self.pub.publish(self.twist)
                self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
                self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
                self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
                self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])
      
    def spwan_ball(self, name):

        file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/tennis_ball/ball_main.sdf'
        srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
        self.get_position()
        ball_pose = Pose()
        ball_pose.position.x = self.object_pose.position.x
        ball_pose.position.y = self.object_pose.position.y
        ball_pose.position.z = self.object_pose.position.z + self.spawn_pos_z

        ball_pose.orientation.x = self.object_pose.orientation.x 
        ball_pose.orientation.y = self.object_pose.orientation.y 
        ball_pose.orientation.z = self.object_pose.orientation.z 
        ball_pose.orientation.w = self.object_pose.orientation.w


        file_xml = open(file_localition)
        xml_string=file_xml.read()

        req = SpawnModelRequest()
        req.model_name = name
        req.model_xml = xml_string
        req.initial_pose = ball_pose

        res = srv_spawn_model(req)

    def set_ball_target(self):

        #self.x_target = (np.random.randint(8, 10) + np.random.rand())
        self.x_target = (np.random.randint(10, 12) + np.random.rand())
        self.y_target = (np.random.randint(-3, 3) + np.random.rand())

        self.get_position()
        
        self.x_error = self.x_target - self.object_pose.position.x
        self.y_error = self.y_target - self.object_pose.position.y
        self.s = np.sqrt(self.x_error**2 + self.y_error**2)

    def throw_ball(self):

        duration = 0.001

        self.set_ball_target()

        self.yaw_z = np.arctan(self.y_error/self.x_error)
        self.ror_matrix = rotation_matrix(self.yaw_z)
        self.vz0 = 9.8 * self.ball_fly_time

        h = (self.object_pose.position.z + self.spawn_pos_z) + self.vz0 * self.ball_fly_time - (9.8 * self.ball_fly_time**2)/2
        self.ball_fly_time_plus = np.sqrt(2 * h / 9.8)
        self.v0 = self.s/(self.ball_fly_time + self.ball_fly_time_plus)

        if self.v0 > 0 :
            self.v0 += 12
        else:
            self.v0 -= 12

        self.v = np.sqrt(self.v0**2 + self.vz0**2)
        self.launch_angle = np.arctan(self.vz0/self.v0)

        self.force = [self.v0 * 0.057 / duration, 0, self.vz0 * 0.057 / duration ]
        
        self.apply_force, self.apply_torque = get_wrench(self.force, self.torque, self.ror_matrix)

        self.ball_apply_force(self.ball_name, self.apply_force, self.apply_torque, duration)

        self.ball_pre_vel_linear_x = self.away_ball_vel.linear.x 
        self.ball_pre_vel_linear_y = self.away_ball_vel.linear.y  

        print("v0, vz0 : ",self.v0, self.vz0)


    def ball_apply_force(self, target, force, torque, duration):
        
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


        self.gat_away_ball_stats()


    def ball_catch_check(self):
        self.gat_away_ball_stats()

        self.get_position()

        distance_x = (self.away_ball_pose.position.x - self.object_pose.position.x)
        distance_y = (self.away_ball_pose.position.y - self.object_pose.position.y)
        distance_z = (self.away_ball_pose.position.z - self.object_pose.position.z)

        #distance = np.sqrt((distance_x)**2 + (distance_y)**2 + (distance_z)**2)

        if (abs(distance_x) < 0.6 and abs(distance_y) < 1  and abs(distance_z) < 2) or abs(self.away_ball_pose.position.x) > 18:
        #if (abs(distance_x) < 0.6 and abs(distance_y) <0.6  and abs(distance_z) < 1):
        
            self.del_ball()
            return  True

        return False

    def del_ball(self):
        srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        res = srv_delete_model(self.away_ball_name)
        self.away_ball_vel_max_x = 0
        self.away_ball_vel_max_y = 0


    def gat_away_ball_stats(self):
        self.ball_state = self.g_get_state(model_name = self.away_ball_name)

        self.away_ball_pose = Pose()
        self.away_ball_pose.position.x = float(self.ball_state.pose.position.x)
        self.away_ball_pose.position.y = float(self.ball_state.pose.position.y)
        self.away_ball_pose.position.z = float(self.ball_state.pose.position.z)
        
        self.away_ball_vel = Twist()

        self.away_ball_vel.linear.x = float(self.ball_state.twist.linear.x)
        self.away_ball_vel.linear.y = float(self.ball_state.twist.linear.y)
        self.away_ball_vel.linear.z = float(self.ball_state.twist.linear.z)

        self.away_ball_vel.angular.x = float(self.ball_state.twist.angular.x)
        self.away_ball_vel.angular.y = float(self.ball_state.twist.angular.y)
        self.away_ball_vel.angular.z = float(self.ball_state.twist.angular.z)

        if self.away_ball_vel.linear.x > self.away_ball_vel_max_x:
            self.away_ball_vel_max_x = self.away_ball_vel.linear.x

        if self.away_ball_vel.linear.y > self.away_ball_vel_max_y:
            self.away_ball_vel_max_y = self.away_ball_vel.linear.y




class Make_mecanum_right(Make_mecanum_left):

    def __init__(self, model_name):
        Make_mecanum_left.__init__(self, model_name)
        
        self.x_move_target = np.nan
        self.y_move_target = np.nan
        self.esti_ball_landing_point = [np.nan]

    def set_ball_target(self):
        #self.x_target = -(np.random.randint(8, 10) + np.random.rand())
        
        self.x_target = -(np.random.randint(12, 15) + np.random.rand())
        self.y_target = (np.random.randint(-3, 3) + np.random.rand())

        self.get_position()
        
        self.x_error = self.x_target - self.object_pose.position.x
        self.y_error = self.y_target - self.object_pose.position.y
        self.s = -np.sqrt(self.x_error**2 + self.y_error**2)

    
    def move(self, x_target, y_target, away_mecanum):
        t0 = time.time()

        while True:
            return_home(away_mecanum)

            if self.ball_catch_check():

                self.stop()
                away_mecanum.stop()
                break 

            self.get_position()

            t1 = time.time()
            self.dt = t1 - t0

            #self.cal_liftdrag()
            t0 = time.time()

            #print("dt : ", self.dt)

            
            self.x_error = self.object_pose.position.x -  x_target 
            self.y_error = self.object_pose.position.y -  y_target
            
            #print(self.x_error, self.y_error)
            if (abs(self.x_error) <0.1 and abs(self.y_error)< 0.1) :
                self.stop()
                away_mecanum.stop()
        
            else:
                self.set_x_velocity(self.dt)
                self.set_y_velocity(self.dt)
                if abs(self.x_error) < 0.1:
                    self.vel_forward_apply = 0

                if abs(self.y_error) < 0.1:
                    self.vel_lateral_apply = 0


                self.twist = Twist()
                #print(self.vel_forward_apply, self.vel_lateral_apply)
                
                self.twist.linear.x = self.vel_forward_apply
                self.twist.linear.y = self.vel_lateral_apply
                self.twist.linear.z = 0

                self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

                self.pub.publish(self.twist)
                self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
                self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
                self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
                self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])

    def callback_landing_point(self, data):

        if len(data.data) < 1 :
            self.esti_ball_landing_point = [np.nan]
            return 0

        self.esti_ball_landing_point = [data.data[0], data.data[1], data.data[2]]
        
        #print(self.esti_ball_landing_point)
    
    def move_base_camera(self, add_catch_point,away_mecanum):
            t0 = time.time()

            while True:

                rospy.Subscriber("esti_landing_point", Float64MultiArray, self.callback_landing_point)

                return_home(away_mecanum)

                if self.ball_catch_check():
                    self.stop()
                    away_mecanum.stop()
                    self.x_move_target = np.nan
                    break 

                self.get_position()

                t1 = time.time()
                self.dt = t1 - t0

                #self.cal_liftdrag()
                t0 = time.time()

                #print("dt : ", self.dt)

                if np.isnan(self.esti_ball_landing_point[0]) == False :
                    self.x_move_target = self.esti_ball_landing_point[0]
                    self.y_move_target = self.esti_ball_landing_point[1]

                if np.isnan(self.x_move_target) == False :

                    self.x_error = self.object_pose.position.x -  (self.x_move_target + add_catch_point) 
                    self.y_error = self.object_pose.position.y -  self.y_move_target
                    
                    #print(self.x_error, self.y_error)
                    if (abs(self.x_error) <0.1 and abs(self.y_error)< 0.1) :
                        self.stop()
                        away_mecanum.stop()
                        self.x_move_target = np.nan

                    else:
                        self.set_x_velocity(self.dt)
                        self.set_y_velocity(self.dt)
                        if abs(self.x_error) < 0.1:
                            self.vel_forward_apply = 0

                        if abs(self.y_error) < 0.1:
                            self.vel_lateral_apply = 0


                        self.twist = Twist()
                        #print(self.vel_forward_apply, self.vel_lateral_apply)
                        
                        self.twist.linear.x = self.vel_forward_apply
                        self.twist.linear.y = self.vel_lateral_apply
                        self.twist.linear.z = 0

                        self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

                        self.pub.publish(self.twist)
                        self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
                        self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
                        self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
                        self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])


def return_home(home_mecanum):

    home_mecanum.get_position()

    robot_x = home_mecanum.object_pose.position.x
    robot_y = home_mecanum.object_pose.position.y
    robot_z = home_mecanum.object_pose.position.z

    robot_angle = np.rad2deg(home_mecanum.angle[2])

    if robot_x < 0:
        x_error = -12 - robot_x
        y_error = -robot_y

        home_mecanum.twist.angular.z = -robot_angle/100

    if robot_x > 0:
        x_error = robot_x - 12
        y_error = robot_y

        if robot_angle > 0 :
            home_mecanum.twist.angular.z = (180 - robot_angle)/100
        else:
            home_mecanum.twist.angular.z = -(180 + robot_angle)/100

    vel_forward_apply, vel_lateral_apply = home_mecanum.check_velocity(home_mecanum.vel_forward * (x_error*0.5), 
                                                                        home_mecanum.vel_lateral * (y_error*0.5))
    
    home_mecanum.twist.linear.x = vel_forward_apply
    home_mecanum.twist.linear.y = vel_lateral_apply
    home_mecanum.twist.linear.z = 0

    home_mecanum.wheel_vel = mecanum_wheel_velocity(home_mecanum.twist.linear.x, home_mecanum.twist.linear.y, home_mecanum.twist.angular.z)

    home_mecanum.pub.publish(home_mecanum.twist)
    home_mecanum.pub_wheel_vel_1.publish(home_mecanum.wheel_vel[0,:])
    home_mecanum.pub_wheel_vel_2.publish(home_mecanum.wheel_vel[1,:])
    home_mecanum.pub_wheel_vel_3.publish(home_mecanum.wheel_vel[2,:])
    home_mecanum.pub_wheel_vel_4.publish(home_mecanum.wheel_vel[3,:])

    if abs(x_error) <0.1 and abs(y_error)< 0.1 :
        home_mecanum.stop()



