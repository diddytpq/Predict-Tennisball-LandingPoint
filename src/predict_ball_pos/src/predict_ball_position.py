#!/usr/bin/env python
from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])

import numpy as np
from sympy import Symbol, solve

import time

import roslib
import rospy
from std_msgs.msg import String, Float64, Float64MultiArray
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox


roslib.load_manifest('ball_trajectory')

# ball_tracking setup
fgbg = cv2.createBackgroundSubtractorMOG2(100, 16, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# yolov5 setup
conf_thres = 0.25
iou_thres=0.45
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
max_det = 2 # maximum detections per image
hide_labels=False,  # hide labels
hide_conf=False,  # hide confidences
line_thickness=3,  # bounding box thickness (pixels)


set_logging()
device = select_device(0)

weights = path + '/weights/best.pt'
img_size = 640

model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(img_size, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

# draw graph setup

point_image = np.zeros([640,640,3], np.uint8) + 255
trajectroy_image = np.zeros([640,640,3], np.uint8) + 255

tennis_court_img = cv2.imread(path + "/images/tennis_court.png")
tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)

real_ball_trajectory_list = []
estimation_ball_trajectory_list = []
esti_ball_landing_point_list = []

save_flag = 0

disappear_cnt = 0

time_list = []

ball_val_list = []
real_ball_val_list = []
esti_ball_val_list = []

a = []
b = []

#kalman filter setup
color = tuple(np.random.randint(low=75, high = 255, size = 3).tolist())


class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        self.landingpoint = [0, 0]

        rospy.init_node('Image_converter', anonymous=True)

        #send topic to landing point check.py
        self.pub = rospy.Publisher('/esti_landing_point',Float64MultiArray, queue_size = 10)
        self.array2data = Float64MultiArray()
        
        rospy.Subscriber("/camera_left_top_ir/depth/image_raw", Image, self.callback_left_top_depth)
        rospy.Subscriber("/camera_left_0/depth/image_raw",Image,self.callback_left_depth_0)
        rospy.Subscriber("/camera_right_0_ir/camera_right_0/color/image_raw",Image,self.callback_right_0)

        rospy.Subscriber("/camera_right_0/depth/image_raw",Image,self.callback_right_depth_0)
        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.callback_left_0)
        rospy.Subscriber("/camera_left_top_ir/camera_left_top_ir/color/image_raw", Image, self.callback_left_top_ir)
        rospy.Subscriber("/camera_right_1_ir/camera_right_1/color/image_raw",Image,self.main)

    def callback_left_top_ir(self, data):
        try:
            self.t0 = time.time()
            self.left_top_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_left_0(self, data):
        try:
            self.t0 = time.time()
            self.left_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        except CvBridgeError as e:
            print(e)

    def callback_right_0(self, data):
        try:
            self.right_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
        except CvBridgeError as e:
            print(e)

    def callback_left_top_depth(self, data):
        try:
            self.left_top_depth_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")

        except CvBridgeError as e:
            print(e)

    def callback_left_depth_0(self, data):
        try:
            self.left_depth_0_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
            
        except CvBridgeError as e:
            print(e)

    def callback_right_depth_0(self, data):
        try:
            self.right_depth_0_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

    def ball_tracking(self, image):

            self.ball_cand_box = []

            image_ori = image.copy()

            self.blur = cv2.GaussianBlur(image_ori, (13, 13), 0)

            self.fgmask_1 = fgbg.apply(self.blur, None, 0.01)

            #self.fgmask_erode = cv2.erode(self.fgmask_1, kernel_erosion_1, iterations = 1) #오픈 연산이아니라 침식으로 바꾸자

            self.fgmask_dila = cv2.dilate(self.fgmask_1,kernel_dilation_2,iterations = 1)

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.fgmask_dila, connectivity = 8)

            for i in range(len(stats)):
                x, y, w, h, area = stats[i]
                
                if area > 3000 : # or area < 500 or aspect > 1.2 or aspect < 0.97 : 
                    continue
                cv2.rectangle(image_ori, (x, y), (x + w, y + h), (255,0,0), 3)

                x0, y0, x1, y1 = x, y, x+w, y+h

                self.ball_cand_box.append([x0, y0, x1, y1 ])
            
            return image_ori

    def robot_tracking(self, image):

            self.robot_box = []

            image_ori = image.copy()
            img = letterbox(image_ori, imgsz, stride=stride)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img_in = torch.from_numpy(img).to(device)
            img_in = img_in.float()
            img_in /= 255.0

            if img_in.ndimension() == 3:
                img_in = img_in.unsqueeze(0)
            

            pred = model(img_in, augment=False)[0]

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):  # detections per image
                
                im0 = image_ori.copy()

                if len(det):
                    det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class

                        label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                        x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        self.robot_box.append([x0, y0, x1, y1])

            return im0

    def check_iou(self, robot_box, ball_cand_box):
        no_ball_box = []
        centroid_ball = []

        if len(robot_box) < 1:
            self.ball_box = ball_cand_box
            return 0

        for i in range(len(robot_box)):
            
            for j in range(len(ball_cand_box)):
                if self.iou(robot_box[i], ball_cand_box[j]):
                    no_ball_box.append(ball_cand_box[j])

        for i in no_ball_box:
            del ball_cand_box[ball_cand_box.index(i)]
        
        self.ball_box = ball_cand_box

    def iou(self,box_0, box_1):
        b0x_0, b0y_0, b0x_1 ,b0y_1 = box_0
        b1x_0, b1y_0, b1x_1 ,b1y_1 = box_1

        min_x = np.argmin([b0x_0,b1x_0])
        min_y = np.argmin([b0y_0,b1y_0])

        if min_x == 0 and min_y == 0:
            if ((b0x_0 <= b1x_0 <= b0x_1) or (b0x_0 <= b1x_1 <= b0x_1)) and ((b0y_0 <= b1y_0 <= b0y_1) or (b0y_0 <= b1y_1 <= b0y_1)):
                return True
        if min_x == 0 and min_y == 1:
            if ((b0x_0 <= b1x_0 <= b0x_1) or (b0x_0 <= b1x_1 <= b0x_1)) and ((b1y_0 <= b0y_0 <= b1y_1) or (b1y_0 <= b0y_1 <= b1y_1)):
                return True
        if min_x == 1 and min_y == 0:
            if ((b1x_0 <= b0x_0 <= b1x_1) or (b1x_0 <= b0x_1 <= b1x_1)) and ((b0y_0 <= b1y_0 <= b0y_1) or (b0y_0 <= b1y_1 <= b0y_1)):
                return True
        if min_x == 1 and min_y == 1:
            if ((b1x_0 <= b0x_0 <= b1x_1) or (b1x_0 <= b0x_1 <= b1x_1) ) and ((b1y_0 <= b0y_0 <= b1y_1) or (b1y_0 <= b0y_1 <= b1y_1) ):
                return True

        return False
    


    def get_depth(self, L_pos, R_pos):
        
        cx = 320
        cy = 160
        focal_length = 343.159
        
        x_L, y_L = L_pos[0] - cx, L_pos[1] - cy
        x_R, y_R = R_pos[0] - cx, R_pos[1] - cy
        
        a_L = np.sqrt(focal_length ** 2 + x_L ** 2 + y_L ** 2)
        b_L = np.sqrt(focal_length ** 2 + x_L ** 2)

        if x_L < 0:
            th_L = 0.785398 + np.arccos(focal_length / b_L)

        else :
            th_L = 0.785398 - np.arccos(focal_length / b_L)


        c_L = b_L * np.cos(th_L)
        
        a_R = np.sqrt(focal_length ** 2 + x_R ** 2 + y_R ** 2)
        b_R = np.sqrt(focal_length ** 2 + x_R ** 2)

        if x_R > 0:
            th_R = 0.785398 + np.arccos(focal_length / b_R)

        else :
            th_R = 0.785398 - np.arccos(focal_length / b_R)

        c_R = b_R * np.cos(th_R)
        
        theta_L = np.arccos(c_L/a_L)
        theta_R = np.arccos(c_R/a_R)
        
        
        D_L = 12.8 * np.sin(theta_R) / np.sin(3.14 - (theta_L + theta_R))
        D_R = 12.8 * np.sin(theta_L) / np.sin(3.14 - (theta_L + theta_R))

        return D_L, D_R

    

    def cal_ball_height(self, ball_distance_list, ball_centroid_list):

        x_pos_L, y_pos_L = ball_centroid_list[0]
        x_pos_R, y_pos_R = ball_centroid_list[1]

        distance_L = ball_distance_list[0]
        distance_R = ball_distance_list[1]


        if y_pos_R > 320:
            y_pos_R -= 320


        #Left camera
        x_pos_L = x_pos_L - 320
        y_pos_L = y_pos_L - 160

        focal2x_point_length = np.sqrt(x_pos_L**2 + 343.159073**2)

        height_theta_pix = np.arctan(abs(y_pos_L) / focal2x_point_length)

        height = distance_L * np.sin(height_theta_pix)

        if y_pos_L < 160:
            height += 1
        
        else:
            1 - height 

        #rigth camera
        #


        
        return height, height

    def cal_ball_position(self, ball_height_list, ball_distance_list):

        height = sum(ball_height_list) / 2 - 1
        
        #삼각형 결정조건 검증
        triangle_length = [ball_distance_list[0], ball_distance_list[1], 12.8]
        triangle_max_length = triangle_length[np.argmax(triangle_length)]
        del triangle_length[np.argmax(triangle_length)]

        if sum(triangle_length) < triangle_max_length :
            return [np.nan, np.nan, np.nan]
        
        th_L = np.arccos((ball_distance_list[0] ** 2 + 12.8 ** 2 - ball_distance_list[1]**2) / (2 * 12.8 * ball_distance_list[0]))
        
        ball2net_length_x_L = ball_distance_list[0] * np.sin(th_L)
        ball_position_y_L = ball_distance_list[0] * np.cos(th_L)

        ball_plate_angle_L = np.sin(height / ball2net_length_x_L)
        
        ball_position_x_L = ball2net_length_x_L * np.cos(ball_plate_angle_L)
        
        #th_R = np.arccos((ball_depth_list[1] ** 2 + 12.8 ** 2 - ball_depth_list[0]**2) / (2 * 12.8 * ball_depth_list[1]))
        
        #ball2net_length_x_R = ball_depth_list[1] * np.sin(th_R)
        #ball_position_y_R = ball_depth_list[1] * np.cos(th_R)
        
        #ball_plate_angle_R = np.sin(height / ball2net_length_x_R)
        
        #ball_position_x_R = ball2net_length_x_R * np.cos(ball_plate_angle_R)

        return [-ball_position_x_L, ball_position_y_L - 6.4, height + 1]

    
    def draw_point_court(self, real_point_list, camera_predict_point_list):

        real_pix_point_list = []
        predict_pix_point_list = []

        if np.isnan(camera_predict_point_list[0]):
            return 0

        x_pred = camera_predict_point_list[0]
        y_pred = camera_predict_point_list[1]

        y_pix_length, x_pix_length = tennis_court_img.shape[0], tennis_court_img.shape[1]

        x_meter2pix = 23.77 / x_pix_length
        y_meter2pix = 10.97 / y_pix_length

        real_pix_point_list.append(int(np.round((11.885 + real_point_list[0]) / x_meter2pix)))
        predict_pix_point_list.append(int(np.round((11.885 + x_pred) / x_meter2pix)))

        real_pix_point_list.append(int(np.round((5.485 - real_point_list[1]) / y_meter2pix)))
        predict_pix_point_list.append(int(np.round((5.485 - y_pred) / y_meter2pix)))

        real_pix_point_xy = real_pix_point_list[0:2]
        predict_pix_point = predict_pix_point_list[0:2]

        cv2.circle(tennis_court_img,real_pix_point_xy, 4, [0, 0, 255], -1)
        cv2.circle(tennis_court_img,predict_pix_point, 4, [0, 255, 0], -1)

    def check_ball_seq(self, disappear_cnt):

        global save_flag

        if np.isnan(self.ball_camera_list[0]):
            disappear_cnt += 1

            if disappear_cnt == 5 :

                if save_flag == 0 :
                    if len(esti_ball_landing_point_list):
                        print("-----------------------")
                        print("send meg : ", esti_ball_landing_point_list[-1])
                        self.array2data.data = esti_ball_landing_point_list[-1]
                        self.pub.publish(self.array2data)
                
                    #print(esti_ball_landing_point_list)
                    save_flag = 1

                    print("real_ball_trajectory_list = np.array(", real_ball_trajectory_list ,")")
                    print("estimation_ball_trajectory_list = np.array(", estimation_ball_trajectory_list,")")

                disappear_cnt = 0
                
                real_ball_trajectory_list.clear()
                estimation_ball_trajectory_list.clear()
                
                esti_ball_landing_point_list.clear()
                time_list.clear()

                
                

        else:
            disappear_cnt = 0
            time_list.append(time.time())

            real_ball_trajectory_list.append(self.real_ball_pos_list)
            estimation_ball_trajectory_list.append([np.round(self.ball_camera_list[0],3), np.round(self.ball_camera_list[1],3), np.round(self.ball_camera_list[2],3)])
            
            if np.isnan(self.esti_ball_landing_point[0]) == False:
                esti_ball_landing_point_list.append(self.esti_ball_landing_point)

            save_flag = 0

        return disappear_cnt

    def cal_ball_val(self):

        if len(time_list) > 1 :

            v0, v1 = np.array(estimation_ball_trajectory_list[-2]), np.array(estimation_ball_trajectory_list[-1])
            dt = time_list[-1] - time_list[-2]
            
            real_v0, real_v1 = np.array(real_ball_trajectory_list[-2]), np.array(real_ball_trajectory_list[-1])

            return (v1 - v0)/dt , (real_v1 - real_v0)/dt

        else:
            return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]

    def get_ball_status(self):
        self.g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.ball_state = self.g_get_state(model_name = 'ball_left')

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

    def cal_landing_point(self, pos, vel):

        t_list = []

        x0, y0, z0 = pos[0], pos[1], pos[2]
        vx, vy, vz = vel[0], vel[1], vel[2]

        a = -((0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vz ** 2 ) / 0.057 + 9.8 / 2 )
        b = vz
        c = z0

        t_list.append((-b + np.sqrt(b ** 2 - 4 * a * c))/(2 * a))
        t_list.append((-b - np.sqrt(b ** 2 - 4 * a * c))/(2 * a))

        t = max(t_list)
        
        x = np.array(x0 + vx * t - (0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vx ** 2 ) * (t ** 2) / 0.057,float)
        y = np.array(y0 + vy * t - (0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vy ** 2 ) * (t ** 2) / 0.057,float)
        z = np.array(z0 + vz * t - ((0.5 * 0.507 * 1.2041 * np.pi * (0.033 ** 2) * vz ** 2 ) / 0.057 + 9.8 / 2) * (t ** 2),float)
        
        return [np.round(x,3), np.round(y,3), np.round(z,3)]

    def main(self, data):

        global point_image

        global color
        global tennis_court_img

        global real_ball_trajectory_list
        global estimation_ball_trajectory_list
        global esti_ball_landing_point_list

        global save_flag

        global time_list

        global disappear_cnt

        (rows,cols,channels) = self.left_data_0.shape

        self.ball_box = []

        self.ball_height_list = [[0], [0]]
        self.ball_centroid_list = [[0, 0],[0, 0]]
        self.ball_distance_list = [[0],[0]]
        self.ball_depth_list = [[0],[0]]
        self.esti_ball_val = [np.nan, np.nan, np.nan]
        self.esti_ball_landing_point = [np.nan, np.nan, np.nan]


        self.get_ball_status()

        self.ball_camera_list = [np.nan, np.nan, np.nan]
        
        if cols > 60 and rows > 60 :
            t1 = time.time()

            self.real_ball_pos_list = [np.round(self.ball_pose.position.x,3), np.round(self.ball_pose.position.y,3), np.round(self.ball_pose.position.z,3)]


            self.left_top_frame = cv2.resize(self.left_top_data_0,(640,640),interpolation = cv2.INTER_AREA)
            self.left_frame = cv2.vconcat([self.left_data_0,self.right_data_0])

            self.main_frame = cv2.hconcat([self.left_frame, self.left_top_frame])

            ball_detect_img = self.main_frame.copy()
            robot_detect_img = self.main_frame.copy()
            
            self.left_top_depth = cv2.resize(self.left_top_depth_ori,(640,640),interpolation = cv2.INTER_AREA)
            
            self.left_depth_ori = cv2.vconcat([self.left_depth_0_ori, self.right_depth_0_ori])
            self.main_depth_ori = cv2.hconcat([self.left_depth_ori, self.left_top_depth])

            self.main_depth = np.float32(self.main_depth_ori)

            self.main_depth_frame = cv2.normalize(self.main_depth, None, 0, 255, cv2.NORM_MINMAX)
            self.main_depth_frame = np.uint8(np.round(self.main_depth_frame))

            robot_detect_img = self.robot_tracking(self.left_frame.copy()) #get robot bbox

            #robot_pos = self.get_robot_pos(self.robot_box)  # robot bbox 3개

            self.ball_tracking(self.left_frame.copy())  #get ball cand bbox list
                   
            if self.ball_cand_box:
                self.check_iou(self.robot_box, self.ball_cand_box) # get ball bbox list

            if self.ball_box:  #draw ball bbox and trajectory and predict ball pos

                for i in range(len(self.ball_box)):
                    x0, y0, x1, y1 = self.ball_box[i]

                    ball_x_pos, ball_y_pos = int((x0 + x1)/2), int((y0 +y1)/2)

                    cv2.rectangle(ball_detect_img, (x0, y0), (x1, y1), color, 3)
                    cv2.circle(point_image,(ball_x_pos, ball_y_pos), 4, color, -1)

                    #predict ball pos
                    #ball_depth = self.get_depth(x0, y0, x1, y1)

                    if ball_x_pos < 640:

                        if ball_y_pos < 320:
                            #self.ball_height_list[0], self.ball_distance_list[0] = self.cal_ball_height(ball_depth, ball_x_pos, ball_y_pos)
                            self.ball_centroid_list[0] = [ball_x_pos, ball_y_pos]

                            #self.ball_depth_list[0] = ball_depth
                            
                        else:
                            #self.ball_height_list[1], self.ball_distance_list[1] = self.cal_ball_height(ball_depth, ball_x_pos, ball_y_pos)
                            self.ball_centroid_list[1] = [ball_x_pos, ball_y_pos - 320]

                            #self.ball_depth_list[1] = ball_depth
                
                self.ball_distance_list[0], self.ball_distance_list[1] = self.get_depth(self.ball_centroid_list[0], self.ball_centroid_list[1])
                self.ball_height_list[0], self.ball_height_list[1] = self.cal_ball_height(self.ball_distance_list, self.ball_centroid_list)

            if min(self.ball_centroid_list) > [0, 0]:
                
                self.ball_camera_list  = self.cal_ball_position(self.ball_height_list, self.ball_distance_list)
                
                if np.isnan(self.ball_camera_list[0]) == False:
                    self.ball_camera_list[0] = self.ball_camera_list[0] + 0.4

                print("------------------------------------------------------------------")
                #print("real_distance : ", np.round(np.sqrt(self.real_ball_pos_list[0] **2 + (self.real_ball_pos_list[1] - (-6.4)) ** 2 + (self.real_ball_pos_list[2] - 1) ** 2), 3), 
                #                         np.round(np.sqrt(self.real_ball_pos_list[0] **2 + (self.real_ball_pos_list[1] - (6.4)) ** 2 + (self.real_ball_pos_list[2] - 1) ** 2), 3))
                #print("distance : ", np.round(self.ball_distance_list[0], 3), np.round(self.ball_distance_list[1], 3))
                
                print("real_ball_pos = [{}, {}, {}]".format(self.real_ball_pos_list[0], self.real_ball_pos_list[1], self.real_ball_pos_list[2]))
                print("camera_preadict_pos = " ,[np.round(self.ball_camera_list[0],3), np.round(self.ball_camera_list[1],3), np.round(self.ball_camera_list[2],3)])

                #a.append([np.round(np.sqrt(self.real_ball_pos_list[0] **2 + (self.real_ball_pos_list[1] - (-6.4)) ** 2 + (self.real_ball_pos_list[2] - 1) ** 2), 3), 
                #                         np.round(np.sqrt(self.real_ball_pos_list[0] **2 + (self.real_ball_pos_list[1] - (6.4)) ** 2 + (self.real_ball_pos_list[2] - 1) ** 2), 3)])

                #b.append([np.round(self.ball_distance_list[0], 3), np.round(self.ball_distance_list[1], 3)])
                #print("real_distance = np.array(",a,")")
                #print("distance = np.array(",b,")")



            self.esti_ball_val, self.real_ball_val = self.cal_ball_val()
            
            if np.isnan(self.ball_camera_list[0]) == False and np.isnan(self.esti_ball_val[0]) == False:

               # print("ball_val = " ,[np.round(self.ball_vel.linear.x,3), np.round(self.ball_vel.linear.y,3), np.round(self.ball_vel.linear.z,3)])
               # print("real_ball_val = " ,[self.real_ball_val[0], self.real_ball_val[1], self.real_ball_val[2]])
               # print("esti_ball_val = " ,[self.esti_ball_val[0], self.esti_ball_val[1], self.esti_ball_val[2]])

                """ball_val_list.append([np.round(self.ball_vel.linear.x,3), np.round(self.ball_vel.linear.y,3), np.round(self.ball_vel.linear.z,3)])
                real_ball_val_list.append([self.real_ball_val[0], self.real_ball_val[1], self.real_ball_val[2]])
                esti_ball_val_list.append([self.esti_ball_val[0], self.esti_ball_val[1], self.esti_ball_val[2]])

                print("ball_val_list = np.array(", ball_val_list , ')')
                print("real_ball_val_list = np.array(", real_ball_val_list , ')')
                print("esti_ball_val_list = np.array(", esti_ball_val_list , ')')"""


                self.esti_ball_landing_point = self.cal_landing_point(self.ball_camera_list, self.esti_ball_val)


            #disappear_cnt = self.check_ball_seq(disappear_cnt)



            self.draw_point_court(self.real_ball_pos_list, self.ball_camera_list)



            #trajectroy_image = cv2.hconcat([point_image[:320,:640,:],point_image[320:,:640,:]])

            t2 = time.time()

            #cv2.imshow("left_frame", self.left_frame)
            #cv2.imshow("main_depth_0", self.main_depth_frame)

            #cv2.imshow("image_robot_tracking", robot_detect_img)
            
            cv2.imshow("ball_detect_img", ball_detect_img)
            cv2.imshow("tennis_court", tennis_court_img)

            #cv2.imshow("trajectroy_image", trajectroy_image)

            #print(1/(t2-t1))
            
            key = cv2.waitKey(1)

            if key == 27 : 
                cv2.destroyAllWindows()

            if key == ord("c") : 
                tennis_court_img = cv2.imread(path + "/images/tennis_court.png")
                tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)





def main(args):

    # srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    # res = srv_delete_model("ball_left")

    ic = Image_converter()

    try:
        rospy.spin()


    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)