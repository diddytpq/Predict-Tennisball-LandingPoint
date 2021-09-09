#!/usr/bin/env python

import numpy as np
import time
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String, Float64, Float64MultiArray
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

path = str(FILE.parents[0])


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox

from kalman_utils.UKFilter import * 



roslib.load_manifest('ball_trajectory')

# ball_trajectroy setup
fgbg = cv2.createBackgroundSubtractorMOG2(100, 16, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
kernel_erosion_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))



# yolov5 setup
conf_thres = 0.25
iou_thres=0.45
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
max_det = 1000 # maximum detections per image
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

camera_L_points = np.float32([[26,198],[318,181],[639,200],[177,316]])
camera_R_points = np.float32([[317,181],[615,198],[442,316],[0,200]])

court_img_L_points = np.float32([[0,600],[0,0],[638,0],[531,600]])
court_img_R_points = np.float32([[0,600],[0,0],[531,0],[638,600]])

h_L = cv2.getPerspectiveTransform(camera_L_points, court_img_L_points)
h_R = cv2.getPerspectiveTransform(camera_R_points, court_img_R_points)

point_image = np.zeros([640,640,3], np.uint8) + 255
trajectroy_image = np.zeros([640,640,3], np.uint8) + 255

tennis_court_img = cv2.imread(path + "/images/tennis_court.png")
tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)

ball_ukf = Trajectory_ukf()

color = tuple(np.random.randint(low=75, high = 255, size = 3).tolist())


class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        self.landingpoint = [0, 0]

        rospy.init_node('Image_converter', anonymous=True)


        rospy.Subscriber("/camera_left_0/depth/image_raw",Image,self.callback_left_depth_0)
        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.callback_left_0)

        #rospy.Subscriber("/camera_left_1/depth/image_raw",Image,self.callback_left_depth_1)
        #rospy.Subscriber("/camera_left_1_ir/camera_left_1/color/image_raw",Image,self.callback_left_1)

        rospy.Subscriber("/camera_right_0/depth/image_raw",Image,self.callback_right_depth_0)
        rospy.Subscriber("/camera_right_0_ir/camera_right_0/color/image_raw",Image,self.callback_right_0)

        #rospy.Subscriber("/camera_right_1/depth/image_raw",Image,self.callback_right_depth_1)
        rospy.Subscriber("/camera_right_1_ir/camera_right_1/color/image_raw",Image,self.callback_right_1)

    
    def landing_point_callback(self, data):
        self.landingpoint = [data.data[0], data.data[1]]

    def callback_left_0(self, data):
        try:
            self.t0 = time.time()
            self.left_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        except CvBridgeError as e:
            print(e)

    def callback_left_1(self, data):
        try:
            self.left_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        except CvBridgeError as e:
            print(e)

    def callback_right_0(self, data):
        try:
            self.right_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        except CvBridgeError as e:
            print(e)

    def callback_right_1(self, data):
        try:
            self.right_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.main()
        
        except CvBridgeError as e:
            print(e)

    def callback_left_depth_0(self, data):
        try:
            self.left_depth_0_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            #self.left_depth_0 = np.uint8(np.round(depth_data))
            
        except CvBridgeError as e:
            print(e)

    def callback_left_depth_1(self, data):
        try:
            self.left_depth_1_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            #self.left_depth_1 = np.round(depth_data) 


        except CvBridgeError as e:
            print(e)

    def callback_right_depth_0(self, data):
        try:
            self.right_depth_0_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)

        except CvBridgeError as e:
            print(e)

    def callback_right_depth_1(self, data):
        try:
            self.right_depth_1_ori =self.bridge.imgmsg_to_cv2(data, "passthrough")
            #depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)

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

    def get_robot_pos(self, robot_bbox):

        robot_pos_list = []
        print("---------------------------")

        for x0, y0, x1, y1 in robot_bbox:

            if y0 < (self.main_frame.shape[0]/2): #left camera robot position
                
                depth = self.get_depth(x0, y0, x1, y1)

                #x_center = (x1 + x0) / 2
                #y_center = (y0 + y1) * (1 - (depth / 20))

                #x_center = x0 
                #y_center = y1 - ((y1 - y0) / 5)

                #print(depth)

                for x_center, y_center in [[x0, y0], [x1, y0] ,[x0, y1], [x1, y1]]:

                    robot_pos_court = h_L @ np.array([x_center, y_center, 1]).reshape(3,1)

                    robot_pos_list.append([robot_pos_court[0]/robot_pos_court[2], robot_pos_court[1]/robot_pos_court[2]])

                    cv2.circle(tennis_court_img, (int(robot_pos_court[0]/robot_pos_court[2]), int(robot_pos_court[1]/robot_pos_court[2])), 4, [0, 0, 255], -1)

            else:                             #right camera robot position
                
                depth = self.get_depth(x0, y0, x1, y1)

                #x_center = (x1 + x0) / 2
                #y_center = (y0 + y1 - 320) * (1 - (depth / 20))
                #x_center = x0 
                #y_center = y1 - 320 - ((y1 - y0) / 5)

                #print(depth)
                for x_center, y_center in [[x0, y0], [x1, y0] ,[x0, y1], [x1, y1]]:

                    y_center = y_center - 320

                    robot_pos_court = h_R @ np.array([x_center, y_center, 1]).reshape(3,1)



                    robot_pos_list.append([robot_pos_court[0]/robot_pos_court[2], robot_pos_court[1]/robot_pos_court[2]])

                    cv2.circle(tennis_court_img, (int(robot_pos_court[0]/robot_pos_court[2]), int(robot_pos_court[1]/robot_pos_court[2])), 4, [0, 212, 125], -1)
                    
                    print((int(robot_pos_court[0]/robot_pos_court[2]), int(robot_pos_court[1]/robot_pos_court[2])))  

        return [np.sum(np.array(robot_pos_list)[:,0])/len(robot_bbox) , np.sum(np.array(robot_pos_list)[:,1])/len(robot_bbox)]


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
    
    def get_depth(self, x0, y0, x1, y1):

        depth_list = self.left_depth[y0 : y1, x0 : x1].flatten()
        depth_list = depth_list[np.isfinite(depth_list)]

        if depth_list.shape[0] != 0:
            ball_depth = np.min(depth_list)

        else:
            return False

        return ball_depth

    def cal_ball_height(self, depth, x_pos, y_pos):

        if y_pos > 320:
            y_pos -= 320

        x = x_pos - 320
        y = y_pos - 160

        focal_length = np.sqrt(x**2 + 337**2)
        theta_pix = np.arctan(abs(y) / focal_length)

        height = depth * np.sin(theta_pix)

        if y_pos < 160:
            height += 1
        
        else:
            1 - height 
        
        return height

    def cal_ball_position(self, ball_height_list, ball_depth_list):

        height = sum(ball_height_list) / 2 - 1
        
        th_L = np.arccos((ball_depth_list[0] ** 2 + 12.8 ** 2 - ball_depth_list[1]**2) / (2 * 12.8 * ball_depth_list[0]))
        #th_R = np.arccos((ball_depth_list[1] ** 2 + 12.8 ** 2 - ball_depth_list[0]**2) / (2 * 12.8 * ball_depth_list[1]))
        
        #print(np.rad2deg(th_L), np.rad2deg(th_R))

        ball2net_length_x_L = ball_depth_list[0] * np.sin(th_L)
        ball_position_y_L = ball_depth_list[0] * np.cos(th_L)

        ball_plate_angle_L = np.sin(height / ball2net_length_x_L)
        
        ball_position_x_L = ball2net_length_x_L * np.cos(ball_plate_angle_L)
        
        #ball2net_length_x_R = ball_depth_list[1] * np.sin(th_R)
        #ball_position_y_R = ball_depth_list[1] * np.cos(th_R)
        
        #ball_plate_angle_R = np.sin(height / ball2net_length_x_R)
        
        #ball_position_x_R = ball2net_length_x_R * np.cos(ball_plate_angle_R)

        return [-ball_position_x_L, ball_position_y_L - 6.4, height + 1]#, [-ball_position_x_R, 6.4 - ball_position_y_R , height + 1]

    
    def draw_point_court(self, real_point_list, camera_predict_point_list, uk_predict_point_list, robot_pos_list):

        real_pix_point_list = []
        predict_pix_point_list = []

        #cv2.circle(tennis_court_img, (int(robot_pos_list[0]), int(robot_pos_list[1])), 4, [0, 212, 255], -1)

        if np.isnan(camera_predict_point_list[0]):
            return 0

        x_pred = uk_predict_point_list[0][0]
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

    def get_uk_pos(self, uk_dir):


        uk_ball_list = []

        if uk_dir:
            for (objectID, pos_list) in uk_dir.items():
                
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                np.random.seed(objectID)

                uk_ball_pos_list = ball_ukf.kf_pred_dict[objectID]

                x_pred, y_pred = uk_ball_pos_list[0], uk_ball_pos_list[1]
                uk_ball_list.append([x_pred, y_pred])

                #x_pred, y_pred, z_pred = uk_ball_pos_list[0], uk_ball_pos_list[1], uk_ball_pos_list[2]

                """print("real ball pos : {}, {}, {}".format(self.real_ball_pos_list[0], self.real_ball_pos_list[1], self.real_ball_pos_list[2]))
                print("uk ball pos : ",x_pred, y_pred)"""
            
        return uk_ball_list



    def main(self):

        global point_image

        global color
        global tennis_court_img

        global ball_ukf

        (rows,cols,channels) = self.left_data_0.shape

        self.ball_box = []

        self.ball_height_list = [[0], [0]]
        self.ball_centroid_list = [[0, 0],[0, 0]]
        self.ball_depth_list = [[0, 0],[0, 0]]

        self.get_ball_status()

        self.ball_camera_list = [np.nan, np.nan, np.nan]
        

        if cols > 60 and rows > 60 :
            t1 = time.time()

            self.real_ball_pos_list = [np.round(self.ball_pose.position.x,3), np.round(self.ball_pose.position.y,3), np.round(self.ball_pose.position.z,3)]

            #self.left_data_0 = cv2.resize(self.left_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.main_frame = cv2.vconcat([self.left_data_0,self.right_data_0])
            self.left_depth = cv2.vconcat([self.left_depth_0_ori, self.right_depth_0_ori])

            self.left_depth = np.float32(self.left_depth)

            self.left_depth_frame = cv2.normalize(self.left_depth, None, 0, 255, cv2.NORM_MINMAX)
            self.left_depth_frame = np.uint8(np.round(self.left_depth_frame))

            image_ori = self.main_frame 
            self.image_robot_tracking = self.robot_tracking(image_ori.copy()) #get robot bbox

            robot_pos = self.get_robot_pos(self.robot_box)

            self.ball_tracking(image_ori.copy())  #get ball cand bbox list
                   
            if self.ball_cand_box:
                self.check_iou(self.robot_box, self.ball_cand_box) # get ball bbox list

            

            if self.ball_box:  #draw ball bbox and trajectory and predict ball pos

                for i in range(len(self.ball_box)):
                    x0, y0, x1, y1 = self.ball_box[i]

                    ball_x_pos, ball_y_pos = int((x0 + x1)/2), int((y0 +y1)/2)

                    cv2.rectangle(image_ori, (x0, y0), (x1, y1), color, 3)
                    cv2.circle(point_image,(ball_x_pos, ball_y_pos), 4, color, -1)

                    #predict ball pos
                    ball_depth = self.get_depth(x0, y0, x1, y1)

                    if ball_y_pos < 320:
                        self.ball_height_list[0] = self.cal_ball_height(ball_depth, ball_x_pos, ball_y_pos)
                        self.ball_centroid_list[0] = [ball_x_pos, ball_y_pos]
                        self.ball_depth_list[0] = ball_depth
                        
                    else:
                        self.ball_height_list[1] = self.cal_ball_height(ball_depth, ball_x_pos, ball_y_pos)
                        self.ball_centroid_list[1] = [ball_x_pos, ball_y_pos - 320]
                        self.ball_depth_list[1] = ball_depth


            if min(self.ball_centroid_list) > [0, 0]:
                
                self.ball_camera_list = self.cal_ball_position(self.ball_height_list, self.ball_depth_list)

                print("------------------------------------------------------------------")
                #print("camera_preadict_pos : ",np.round(self.ball_camera_list[0],3), np.round(self.ball_camera_list[1],3), np.round(self.ball_camera_list[2],3))

                #print("height : ",self.ball_height_list)
                #print("real_depth : ", np.round(np.sqrt(real_ball_pos_list[0] **2 + (real_ball_pos_list[1] - (-6.4)) ** 2 + (real_ball_pos_list[2] - 1) ** 2), 3), np.round(np.sqrt(real_ball_pos_list[0] **2 + (real_ball_pos_list[1] - (6.4)) ** 2 + (real_ball_pos_list[2] - 1) ** 2), 3))
                #print("depth : ", np.round(self.ball_depth_list[0], 3), np.round(self.ball_depth_list[1], 3))
                #print(np.round(ball_x_R,3), np.round(ball_y_R,3), np.round(ball_z_R,3))
                
                

            if np.isnan(self.ball_camera_list[0]):

                uk_dir = ball_ukf.update([])

            else:
                
                uk_dir = ball_ukf.update([self.ball_camera_list[0], self.ball_camera_list[0]])
                #uk_dir = ball_ukf.update([ball_x_L, ball_y_L, ball_z_L])


            self.uk_ball_list = self.get_uk_pos(uk_dir)

            self.draw_point_court(self.real_ball_pos_list, self.ball_camera_list, self.uk_ball_list, robot_pos)

            robot_tracking_img = cv2.hconcat([self.image_robot_tracking[:320,:640,:],self.image_robot_tracking[320:,:640,:]])
            ball_detect_img = cv2.hconcat([image_ori[:320,:640,:],image_ori[320:,:640,:]])
            
            #trajectroy_image = cv2.hconcat([point_image[:320,:640,:],point_image[320:,:640,:]])
            #left_depth_img = cv2.hconcat([self.left_depth_frame[:320,:640], self.left_depth_frame[320:,:640]])



            t2 = time.time()

            #cv2.imshow("left_frame", self.left_frame)
            #cv2.imshow("right_frame", self.right_frame)
            #cv2.imshow("left_depth_0", left_depth_img)

            cv2.imshow("robot_tracking_img", robot_tracking_img)
            #cv2.imshow("image_ori", image_ori)
            
            cv2.imshow("ball_detect_img", ball_detect_img)
            cv2.imshow("tennis_court", tennis_court_img)

            #cv2.imshow("trajectroy_image", trajectroy_image)

            #cv2.imshow("test", image_ori[320:,:640,:])

            #print(1/(t2-t1))
            
            key = cv2.waitKey(1)


            if key == 27 : 
                cv2.destroyAllWindows()

            if key == ord("c") : 
                tennis_court_img = cv2.imread(path + "/images/tennis_court.png")
                tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)


















def main(args):

    ic = Image_converter()

    try:
        rospy.spin()


    except KeyboardInterrupt:
        print("Shutting down")
        f.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)