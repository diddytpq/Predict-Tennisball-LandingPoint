#! /home/drcl_yang/anaconda3/envs/py36/bin/python


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


g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)


left_ball_state= []
right_ball_state= []

pre_z = np.nan
pre_gradient = np.nan

t0 = time.time()
t1 = time.time()
dt = t0 - t1

def check_ball_bounce(cnt_z):
    global pre_z
    global pre_gradient

    if np.isnan(pre_z):
        
        pre_z = cnt_z

        return False

    cnt_gradient = cnt_z - pre_z

    if np.isnan(pre_gradient):

        pre_gradient = cnt_gradient

        return False

    if (cnt_gradient > 0 )== True and (pre_gradient < 0) == False:
        
        pre_gradient = cnt_gradient
        
        return True
    else:
        pre_gradient = cnt_gradient

        return False



def gat_ball_stats(ball_name):
    ball_state = g_get_state(model_name = ball_name)

    """object_pose.position.x = float(robot_state.pose.position.x)
    object_pose.position.y = float(robot_state.pose.position.y)
    object_pose.position.z = float(robot_state.pose.position.z)
    object_pose.orientation.x = float(robot_state.pose.orientation.x)
    object_pose.orientation.y = float(robot_state.pose.orientation.y)
    object_pose.orientation.z = float(robot_state.pose.orientation.z)
    object_pose.orientation.w = float(robot_state.pose.orientation.w)
    
    angle = qua2eular(object_pose.orientation.x, object_pose.orientation.y,
                        object_pose.orientation.z, object_pose.orientation.w)"""

    return ball_state

def check_ball_exist(ball_name_list):

    global t0

    t0 = time.time()
    try:
        if g_get_state(model_name = ball_name_list[0]).success:
            return 1
        
        elif g_get_state(model_name = ball_name_list[1]).success:
            return 2

        else:
            return 0
        
    except rospy.ROSInterruptException:
        return 0


def cal_drag_lift_force(down_motion, drag_force, lift_force, angle_xy, angle_x, cl):

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



    return liftdrag_force_x, liftdrag_force_y, liftdrag_force_z

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



def ball_apply_airdyanmic(ball_name, ball_check_flag):
    global left_ball_state
    global right_ball_state
    global t0, t1, dt

    t1 = time.time()

    dt = 1/100

    dt_gain = 1

    #print(dt)

    if ball_check_flag == 1:
        left_ball_state = gat_ball_stats(ball_name)

        down_motion = 0

        left_ball_state_xy = np.sqrt((left_ball_state.twist.linear.x ** 2) + (left_ball_state.twist.linear.y ** 2))
        left_ball_state_xyz =  np.sqrt((left_ball_state.twist.linear.x ** 2) + (left_ball_state.twist.linear.y ** 2 + (left_ball_state.twist.linear.z ** 2)))

        if left_ball_state.twist.angular.y > 0:
            left_ball_angular_xy = np.sqrt((left_ball_state.twist.angular.x ** 2) + (left_ball_state.twist.angular.y ** 2))

        else:
            left_ball_angular_xy = -np.sqrt((left_ball_state.twist.angular.x ** 2) + (left_ball_state.twist.angular.y ** 2))

        if left_ball_state.twist.linear.x == 0 :
            return 0

        angle_x = np.arctan(left_ball_state.twist.linear.y/left_ball_state.twist.linear.x)
        angle_xy = np.arctan(left_ball_state.twist.linear.z/left_ball_state_xy)

        cd = 0.507
        cl = - 0.75 * 0.033 * left_ball_angular_xy / left_ball_state_xy

        if cl < -0.4:
            cl = -0.4
            return 0

        drag_force = -0.5 * cd * 1.2041 * np.pi * (0.033 ** 2) * left_ball_state_xyz
        lift_force = 0.5 * cl * 1.2041 * np.pi * (0.033 ** 2) * left_ball_state_xyz

        if left_ball_state.twist.linear.z < 0:
            down_motion = 1


        liftdrag_force_x, liftdrag_force_y, liftdrag_force_z = cal_drag_lift_force(down_motion, drag_force, lift_force, angle_xy, angle_x, cl)

        force = [np.round(liftdrag_force_x,5) / (dt * dt_gain), np.round(liftdrag_force_y,5) / (dt * (dt_gain + 1) ),np.round(liftdrag_force_z,5) / (dt * dt_gain)]

    if ball_check_flag == 2:
        right_ball_state = gat_ball_stats(ball_name)

        down_motion = 0

        right_ball_state_xy = np.sqrt((right_ball_state.twist.linear.x ** 2) + (right_ball_state.twist.linear.y ** 2))
        right_ball_state_xyz =  np.sqrt((right_ball_state.twist.linear.x ** 2) + (right_ball_state.twist.linear.y ** 2 + (right_ball_state.twist.linear.z ** 2)))

        if right_ball_state.twist.angular.y > 0:
            right_ball_angular_xy = np.sqrt((right_ball_state.twist.angular.x ** 2) + (right_ball_state.twist.angular.y ** 2))

        else:
            right_ball_angular_xy = -np.sqrt((right_ball_state.twist.angular.x ** 2) + (right_ball_state.twist.angular.y ** 2))

        if right_ball_state.twist.linear.x == 0 :
            return 0

        angle_x = np.arctan(right_ball_state.twist.linear.y/right_ball_state.twist.linear.x)
        angle_xy = np.arctan(right_ball_state.twist.linear.z/right_ball_state_xy)
        
        cd = 0.507
        cl =  0.75 * 0.033 * right_ball_angular_xy / right_ball_state_xy

        if cl < -0.4:
            cl = -0.4
            return 0


        drag_force = -0.5 * cd * 1.2041 * np.pi * (0.033 ** 2) * right_ball_state_xyz
        lift_force = 0.5 * cl * 1.2041 * np.pi * (0.033 ** 2) * right_ball_state_xyz
        
        if right_ball_state.twist.linear.z < 0:
            down_motion = 1


        liftdrag_force_x, liftdrag_force_y, liftdrag_force_z = cal_drag_lift_force(down_motion, drag_force, lift_force, angle_xy, angle_x, cl)

        force = [-np.round(liftdrag_force_x,5) / (dt * dt_gain), -np.round(liftdrag_force_y,5) / (dt * (dt_gain + 1)), np.round(liftdrag_force_z,5) / (dt * dt_gain)]
    
    
    #force = [force[0],0,force[2]]

    ball_apply_force(ball_name, force, [0,0,0], dt)



def main(args):

    time.sleep(3)

    rospy.init_node('ball_airdynamic', anonymous=True)

    rate = rospy.Rate(100)

    rospy.loginfo("###########################################################################") 
    rospy.loginfo("start ball air dynamic plugin") 

    ball_left_name = 'ball_left'
    ball_right_name = 'ball_right'

    ball_name_list = [ball_left_name, ball_right_name]

    try:
        while True:
            t0 = time.time()
            ball_check_flag = check_ball_exist(ball_name_list)


            if ball_check_flag:
                ball_apply_airdyanmic(ball_name_list[ball_check_flag - 1], ball_check_flag)

            rate.sleep()

            print(time.time() - t0)
    except KeyboardInterrupt:
        print("Shutting down")





if __name__ == '__main__':
    main(sys.argv)















# left


"""    def break_ball_rolling(self):
        self.gat_away_ball_stats()
        self.ball_pre_vel_linear_x = self.away_ball_vel.linear.x 
        self.ball_pre_vel_linear_y = self.away_ball_vel.linear.y
        if self.check_bounce() and self.away_ball_pose.position.z < 0.021 :
            self.gat_away_ball_stats()
            w_y2 = self.away_ball_vel.angular.y - 1.5 * 0.033 * (self.away_ball_vel.linear.x - self.away_ball_vel_max_x) / 0.03 ** 2
            w_x2 = self.away_ball_vel.angular.x - 1.5 * 0.033 * (self.away_ball_vel.linear.y - self.away_ball_vel_max_y) / 0.03 ** 2
            self.away_ball_vel_max_x = self.away_ball_vel.linear.x
            self.away_ball_vel_max_y = self.away_ball_vel.linear.y
            force = [0, 0, 0]
            self.apply_torque = [(self.away_ball_vel.angular.x  - w_x2) * 1000, (self.away_ball_vel.angular.y - w_y2) * 1000, 0]
        
            #self.ball_apply_force(self.away_ball_name, force, self.apply_torque, self.duration)
    def check_bounce(self):
        self.gat_away_ball_stats()
        self.current_gradient = self.away_ball_pose.position.z - self.ball_preposition_list_z[-1]
        if self.check_gradient(self.pre_gradient_z[-1]) == False and self.check_gradient(self.current_gradient) == True:
            self.ball_preposition_list_z.append(self.away_ball_pose.position.z)
            self.pre_gradient_z.append(self.current_gradient)
            return True
        else:
            self.ball_preposition_list_z.append(self.away_ball_pose.position.z) 
            self.pre_gradient_z.append(self.current_gradient)
            return False
        
    def check_gradient(self, gradient): 
        if gradient < 0: 
            return False
        
        else: 
            return True
"""





"""    #right
    def break_ball_rolling(self):
    self.gat_away_ball_stats()
    duration = 0.001
    self.ball_pre_vel_linear_x = self.away_ball_vel.linear.x 
    self.ball_pre_vel_linear_y = self.away_ball_vel.linear.y
    if self.check_bounce() and self.away_ball_pose.position.z < 0.021 :
        self.gat_away_ball_stats()
        w_y2 = self.away_ball_vel.angular.y - 1.5 * 0.033 * (self.away_ball_vel.linear.x - self.away_ball_vel_max_x) / 0.03 ** 2
        w_x2 = self.away_ball_vel.angular.x - 1.5 * 0.033 * (self.away_ball_vel.linear.y - self.away_ball_vel_max_y) / 0.03 ** 2
        self.away_ball_vel_max_x = self.away_ball_vel.linear.x
        self.away_ball_vel_max_y = self.away_ball_vel.linear.y
        force = [0, 0, 0]
        self.apply_torque = [-(self.away_ball_vel.angular.x  - w_x2) * 1000, -(self.away_ball_vel.angular.y - w_y2) * 1000, 0]
    
        #self.ball_apply_force(self.away_ball_name, force, self.apply_torque, duration)
"""