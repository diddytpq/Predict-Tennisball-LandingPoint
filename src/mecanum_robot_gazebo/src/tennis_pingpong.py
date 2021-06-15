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

from tool.utils import *
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

        self.ball_fly_time = 1.2 #sec

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
           
            self.vel_forward_apply, self.vel_lateral_apply = self.check_velocity(self.vel_forward * self.x_error, self.vel_lateral * self.y_error)
            
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

        srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        req = DeleteModelRequest()
        req.model_name = "ball" 

        res = srv_delete_model("ball")

        time.sleep(0.2)

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
        
        x_target = (np.random.randint(6, 10) + np.random.rand())
        y_target = (np.random.randint(-3, 3) + np.random.rand())
        
        self.get_position()

        x_error = x_target - self.object_pose.position.x
        y_error = y_target - self.object_pose.position.y

        s = np.sqrt(x_error**2 + y_error**2)

        print(s)

        yaw_z = np.tan(y_error/x_error)
        ror_matrix = rotation_matrix(yaw_z)

class Make_mecanum_right(Make_mecanum_left):

    
    def move(self, x_target, y_target):
        
        while True:
            self.get_position()

            self.x_error = self.object_pose.position.x - x_target
            self.y_error = self.object_pose.position.y - y_target
           
            self.vel_forward_apply, self.vel_lateral_apply = self.check_velocity(self.vel_forward * self.x_error, self.vel_lateral * self.y_error)

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

            srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            req = DeleteModelRequest()
            req.model_name = "ball" 

            res = srv_delete_model("ball")

            time.sleep(0.2)

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
        
        x_target = -(np.random.randint(6, 10) + np.random.rand())
        y_target = (np.random.randint(-3, 3) + np.random.rand())
        
        self.get_position()

        x_error = x_target - self.object_pose.position.x
        y_error = y_target - self.object_pose.position.y
        
        print(s)

        yaw_z = np.tan(y_error/x_error)
        ror_matrix = rotation_matrix(yaw_z)
def mecanum_wheel_velocity(vx, vy, wz):
    r = 0.0762 # radius of wheel
    l = 0.23 #length between {b} and wheel
    w = 0.25225 #depth between {b} abd wheel
    alpha = l + w
    
    q_dot = np.array([wz, vx, vy])
    J_pseudo = np.array([[-alpha, 1, -1],[alpha, 1, 1],[alpha, 1, -1],[alpha, 1,1]])

    u = 1/r * J_pseudo @ np.reshape(q_dot,(3,1))#q_dot.T

    return u


if __name__ == '__main__' :

    rospy.init_node('pingpong')

    while True :
        mecanum_0 = Make_mecanum_left('mecanum_0')
        mecanum_0.move(-11.7, 4)
        mecanum_0.throw_ball()
        mecanum_0.move(-5.5, -4)
        mecanum_0.throw_ball()

        mecanum_0.move(-11.7, -4)
        mecanum_0.throw_ball()

        mecanum_0.move(-5.5, 4)
        mecanum_0.throw_ball()
        
        #mecanum_0.spwan_ball()
        #mecanum_1 = Make_mecanum_right('mecanum_1')
        #mecanum_1.move(10,0)
        #mecanum_1.spwan_ball()



