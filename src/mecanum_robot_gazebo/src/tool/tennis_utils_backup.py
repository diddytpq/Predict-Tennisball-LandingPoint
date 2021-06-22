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
        self.vel_lateral = 3.3 #m/s

        self.vel_forward_apply = 0
        self.vel_lateral_apply = 0


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
   
    def stop(self):
        self.vel_forward_apply = 0
        self.vel_lateral_apply = 0
        self.twist = Twist()

        self.twist.linear.x = self.vel_forward_apply
        self.twist.linear.y = self.vel_lateral_apply
        self.twist.linear.z = 0

        self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)
        self.pub.publish(self.twist)
        self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
        self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
        self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
        self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])
    
    def check_velocity(self, x_vel, y_vel):

        if self.vel_forward < abs(x_vel):
            if x_vel > 0: x_vel = self.vel_forward
            else: x_vel = -self.vel_forward

        if self.vel_lateral < abs(y_vel):
            if y_vel > 0: y_vel = self.vel_lateral
            else: y_vel = -self.vel_lateral
            
        return x_vel, y_vel

    def move(self, x_target, y_target):
        
        while True:
            self.get_position()

            self.x_error = x_target - self.object_pose.position.x
            self.y_error = y_target - self.object_pose.position.y
           
            self.vel_forward_apply, self.vel_lateral_apply = self.check_velocity(self.vel_forward * (self.x_error*2), 
                                                                                    self.vel_lateral * (self.y_error*2))
            
            self.twist = Twist()
            
            self.twist.linear.x = self.vel_forward_apply
            self.twist.linear.y = self.vel_lateral_apply
            self.twist.linear.z = 0

            self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

            self.pub.publish(self.twist)
            self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
            self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
            self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
            self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])

            if abs(self.x_error) <0.1 and abs(self.y_error)< 0.1 :
                self.stop()
                break 

    def spwan_ball(self):

        self.del_ball()

        file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball_main.sdf'
        srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
        self.get_position()
        ball_pose = Pose()
        ball_pose.position.x = self.object_pose.position.x
        ball_pose.position.y = self.object_pose.position.y
        ball_pose.position.z = self.object_pose.position.z + 0.5

        ball_pose.orientation.x = 0
        ball_pose.orientation.y = 0
        ball_pose.orientation.z = 0
        ball_pose.orientation.w = 1


        file_xml = open(file_localition)
        xml_string=file_xml.read()

        req = SpawnModelRequest()
        req.model_name = "ball"  
        req.model_xml = xml_string
        req.initial_pose = ball_pose

        res = srv_spawn_model(req)

    def throw_ball(self):

        duration = 0.01
        self.ball_fly_time = 0.6 #max height time [sec]
        
        self.x_target = (np.random.randint(6, 10) + np.random.rand())
        self.y_target = (np.random.randint(-3, 3) + np.random.rand())


        self.get_position()

        x_error = self.x_target - self.object_pose.position.x
        y_error = self.y_target - self.object_pose.position.y

        yaw_z = np.tan(y_error/x_error)
        ror_matrix = rotation_matrix(yaw_z)
        s = np.sqrt(x_error**2 + y_error**2)
        vz0 = 9.8 * self.ball_fly_time

        h = (self.object_pose.position.z + 0.5) + vz0 * self.ball_fly_time - (9.8 * self.ball_fly_time**2)/2
        self.ball_fly_time_plus = np.sqrt(2 * h / 9.8)
        v0 = s/(self.ball_fly_time + self.ball_fly_time_plus)

        v = np.sqrt(v0**2 + vz0**2)
        launch_angle = np.arctan(vz0/v0)

        force = [v0 * 0.057 * 100, 0, vz0 * 0.057 *100 ]
        torque = [0, 10000, 0]

        rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=10)

        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        body_name = 'ball::base_link'


        wrench = Wrench()
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

        
        """print("----------------------------------------------------")
        v0, rpm = cal(force, torque)
        print("\tx_target : ",self.x_target)
        print("\ty_target : ",self.y_target)
        print("\tx_error, y_error :",x_error,y_error)
        print("\ts : ",s)

        print('\tv0: {} \t RPM: {}' .format(v, rpm))
        print('\tlaunch angle: ',np.rad2deg(launch_angle))
        print('\tForce [N]: ', force)
        print('\tTorque [Nm]: ', torque)
        print('\tvo= : ', force[0]/0.057/100,force[1]/0.057/100,force[2]/0.057/100)"""

    def del_ball(self):
        srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        req = DeleteModelRequest()
        req.model_name = "ball" 

        res = srv_delete_model("ball")
        time.sleep(0.1)

    

class Make_mecanum_right(Make_mecanum_left):

    
    def move(self, x_target, y_target):
        
        while True:
            self.get_position()

            self.x_error = self.object_pose.position.x - x_target
            self.y_error = self.object_pose.position.y - y_target
           
            self.vel_forward_apply, self.vel_lateral_apply = self.check_velocity(self.vel_forward * (self.x_error*2), 
                                                                                    self.vel_lateral * (self.y_error*2))

            self.twist = Twist()
            
            self.twist.linear.x = self.vel_forward_apply
            self.twist.linear.y = self.vel_lateral_apply
            self.twist.linear.z = 0

            self.wheel_vel = mecanum_wheel_velocity(self.twist.linear.x, self.twist.linear.y, self.twist.angular.z)

            self.pub.publish(self.twist)
            self.pub_wheel_vel_1.publish(self.wheel_vel[0,:])
            self.pub_wheel_vel_2.publish(self.wheel_vel[1,:])
            self.pub_wheel_vel_3.publish(self.wheel_vel[2,:])
            self.pub_wheel_vel_4.publish(self.wheel_vel[3,:])

            if abs(self.x_error) <0.1 and abs(self.y_error)< 0.1 :
                self.stop()
                break 


        def spwan_ball(self):

            self.del_ball()
   
            file_localition = roslib.packages.get_pkg_dir('ball_trajectory') + '/urdf/ball_main.sdf'
            srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
            self.get_position()
            ball_pose = Pose()
            ball_pose.position.x = self.object_pose.position.x
            ball_pose.position.y = self.object_pose.position.y
            ball_pose.position.z = self.object_pose.position.z + 0.5

            ball_pose.orientation.x = 0
            ball_pose.orientation.y = 0
            ball_pose.orientation.z = 1
            ball_pose.orientation.w = 0


            file_xml = open(file_localition)
            xml_string=file_xml.read()

            req = SpawnModelRequest()
            req.model_name = "ball"  
            req.model_xml = xml_string
            req.initial_pose = ball_pose

            res = srv_spawn_model(req)

            
    def throw_ball(self):

        duration = 0.01
        self.ball_fly_time = 0.6 #max height time [sec]
        self.x_target = -(np.random.randint(6, 10) + np.random.rand())
        self.y_target = (np.random.randint(-3, 3) + np.random.rand())
        
        self.get_position()

        x_error = self.x_target - self.object_pose.position.x
        y_error = self.y_target - self.object_pose.position.y
        
        yaw_z = np.tan(y_error/x_error)
        ror_matrix = rotation_matrix(yaw_z)

        s = -np.sqrt(x_error**2 + y_error**2)
        vz0 = 9.8 * self.ball_fly_time

        h = (self.object_pose.position.z + 0.5) + vz0 * self.ball_fly_time - (9.8 * self.ball_fly_time**2)/2
        self.ball_fly_time_plus = np.sqrt(2 * h / 9.8)
        v0 = s/(self.ball_fly_time + self.ball_fly_time_plus)

        v = np.sqrt(v0**2 + vz0**2)
        launch_angle = np.arctan(vz0/v0)

        force = [v0 * 0.057 * 100, 0, vz0 * 0.057 *100 ]
        torque = [0, -10000, 0]

        rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=10)

        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        body_name = 'ball::base_link'


        wrench = Wrench()
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

        
        """print("----------------------------------------------------")
        v0, rpm = cal(force, torque)
        print("\tx_target : ",self.x_target)
        print("\ty_target : ",self.y_target)
        print("\tx_error, y_error :",x_error,y_error)
        print("\ts : ",s)

        print('\tv0: {} \t RPM: {}' .format(v, rpm))
        print('\tlaunch angle: ',np.rad2deg(launch_angle))
        print('\tForce [N]: ', force)
        print('\tTorque [Nm]: ', torque)
        print('\tvo= : ', force[0]/0.057/100,force[1]/0.057/100,force[2]/0.057/100)"""






def Ttorpm(T):
    
    r = 0.065
    v0 = T * 0.01


    return v0 * 9.549297


def cal(force, T):
    a = np.sqrt((force[0])**2 + (force[2])**2) / 0.057
    v0 = a *0.01

    rpm = Ttorpm(T[1])

    return v0 , rpm

    

def rotation_matrix(angle):

    return np.array([[np.cos(angle),-np.sin(angle),0],
             [np.sin(angle),np.cos(angle),0],
             [0,0,1]])

def get_wrench(force, torque, matrix):

    F = matrix@(np.array(force).reshape([3,1]))
    T = matrix@(np.array(torque).reshape([3,1]))
    
    F = F.reshape([1,3]).tolist()
    T = T.reshape([1,3]).tolist()


    return F[0], T[0]

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


def mecanum_wheel_velocity(vx, vy, wz):
    r = 0.0762 # radius of wheel
    l = 0.23 #length between {b} and wheel
    w = 0.25225 #depth between {b} abd wheel
    alpha = l + w
    
    q_dot = np.array([wz, vx, vy])
    J_pseudo = np.array([[-alpha, 1, -1],[alpha, 1, 1],[alpha, 1, -1],[alpha, 1,1]])

    u = 1/r * J_pseudo @ np.reshape(q_dot,(3,1))#q_dot.T

    return u

def check_catch(mecanum):

    g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

    while True:
        ball_state = g_get_state(model_name='ball')
        mecanum.get_position()

        ball_x = ball_state.pose.position.x
        ball_y = ball_state.pose.position.y
        ball_z = ball_state.pose.position.z
        
        robot_x = mecanum.object_pose.position.x
        robot_y = mecanum.object_pose.position.y
        robot_z = mecanum.object_pose.position.z

        distance = np.sqrt((robot_x - ball_x)**2 + (robot_y - ball_y)**2 + (robot_z - ball_z)**2)
        print(distance)

        if distance< 0.6 or distance > 20:
            mecanum.del_ball()
            break


