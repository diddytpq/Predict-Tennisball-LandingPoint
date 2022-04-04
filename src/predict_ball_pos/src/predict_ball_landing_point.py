#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])
sys.path.append(path + '/lib')

import numpy as np
import time
import cv2

import roslib
import rospy
from std_msgs.msg import String, Float64, Float64MultiArray
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import torch
import torch.backends.cudnn as cudnn

from tools import *

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

from yolov5.utils.augmentations import letterbox

from kalman_utils.KFilter import *

import ray

device = 0
weights = path + "/lib/yolov5/weights/best.pt"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
classes = [0, 38, 80]
agnostic_nms = False
max_det = 1000
half=False
dnn = False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size


half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()

cudnn.benchmark = True  # set True to speed up constant image size inference


color = tuple(np.random.randint(low=200, high = 255, size = 3).tolist())
color = tuple([0,125,255])

tennis_court_img = cv2.imread(path + "/images/tennis_court.png")
tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA) # 1276,600,0

padding_y = int((810 - tennis_court_img.shape[0]) /2 )
padding_x = int((1500 - tennis_court_img.shape[1]) /3)

WHITE = [255,255,255]
tennis_court_img= cv2.copyMakeBorder(tennis_court_img.copy(),padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT,value=WHITE)


disappear_cnt = 0
ball_pos_jrajectory = []

estimation_ball = Ball_Pos_Estimation()

record = False


camera_data = []
camera_depth_data = []


def person_tracking(model, img, img_ori, device):

        person_box_left = []
        person_box_right = []

        img_in = torch.from_numpy(img).to(device)
        img_in = img_in.float()
        img_in /= 255.0

        if img_in.ndimension() == 3:
            img_in = img_in.unsqueeze(0)
        
        pred = model(img_in, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # detections per image
            
            im0 = img_ori.copy()

            if len(det):
                det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    label = names[c] #None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                    x0, y0, x1, y1 = int(xyxy[0]) - 10, int(xyxy[1]) - 10, int(xyxy[2]) + 10, int(xyxy[3] + 10)

                    if y0 < (img_ori.shape[0] / 2) :
                        person_box_left.append([x0, y0, x1, y1])

                    else : 
                        person_box_right.append([x0, y0, x1, y1])
            
        return im0, person_box_left, person_box_right

@ray.remote
class Img_Buffer(object):

    def __init__(self):
        rospy.init_node('Img_Buffer', anonymous=True)

        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image, self.callback_camera_left)
        rospy.Subscriber("/camera_right_0_ir/camera_right_0/color/image_raw",Image, self.callback_camera_right)

        self.img_data_left = []
        self.img_data_right = []
        self.bridge = CvBridge()

    def callback_camera_left(self, data):
        try:
            self.img_data_left = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)


    def callback_camera_right(self, data):
        try:
            self.img_data_right = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def get_img(self):

        return self.img_data_left, self.img_data_right



def get_ball_status():
        t0 = time.time()
        g_get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        ball_state = g_get_state(model_name = 'ball_left')

        ball_pose = Pose()
        ball_pose.position.x = float(ball_state.pose.position.x)
        ball_pose.position.y = float(ball_state.pose.position.y)
        ball_pose.position.z = float(ball_state.pose.position.z)
        
        ball_vel = Twist()

        ball_vel.linear.x = float(ball_state.twist.linear.x)
        ball_vel.linear.y = float(ball_state.twist.linear.y)
        ball_vel.linear.z = float(ball_state.twist.linear.z)

        ball_vel.angular.x = float(ball_state.twist.angular.x)
        ball_vel.angular.y = float(ball_state.twist.angular.y)
        ball_vel.angular.z = float(ball_state.twist.angular.z)

        return ball_pose.position.x, ball_pose.position.y, ball_pose.position.z, ball_vel.linear.x, ball_vel.linear.y, ball_vel.linear.z

def main():
    #global camera_data, camera_depth_data

    global estimation_ball, disappear_cnt, padding_x, padding_y
    global tennis_court_img, ball_pos_jrajectory

    img_buffer = Img_Buffer.remote()

    dT = 1 / 30
    fps = 30

    if record:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("ball_tracking.mp4", codec, fps, (720,810))

    rospy.init_node('Predict_ball_landing_point', anonymous=True)
    pub = rospy.Publisher('/esti_landing_point',Float64MultiArray, queue_size = 10)
    array2data = Float64MultiArray()

    while True:

        camera_data = ray.get(img_buffer.get_img.remote())

        if len(camera_data[0]):

            print("-----------------------------------------------------------------")
            t1 = time.time()
            ball_pose_x, ball_pose_y, ball_pose_z, ball_vel_x, ball_vel_y, ball_vel_z = get_ball_status()

            
            frame_left = camera_data[0]
            frame_right = camera_data[1]

            frame_main = cv2.vconcat([frame_left,frame_right])

            frame = cv2.resize(frame_main, dsize=(640, 720), interpolation=cv2.INTER_LINEAR)

            frame_main = frame

            frame_record = frame_main.copy()

            ball_box_left = []
            ball_box_right = []

            ball_cen_left = []
            ball_cen_right = []

            ball_pos = []

            frame_mog2 = frame_main.copy()
            frame_yolo_main = frame_main.copy()

            img, img_ori = img_preprocessing(frame_yolo_main, imgsz, stride, pt)

            try:
                person_tracking_img, person_box_left_list, person_box_right_list = person_tracking(model, img, img_ori, device)

            except:
                continue
            ball_detect_img, ball_cand_box_list_left, ball_cand_box_list_right = ball_tracking(frame_mog2)  #get ball cand bbox list

            if ball_cand_box_list_left:
                ball_box_left = check_iou(person_box_left_list, ball_cand_box_list_left) # get left camera ball bbox list

            if ball_cand_box_list_right:
                ball_box_right = check_iou(person_box_right_list, ball_cand_box_list_right) # get right camera ball bbox list

            ball_box = [ball_box_left, ball_box_right]

            if ball_box:  #draw ball bbox 
                
                total_ball_box = ball_box[0] + ball_box[1]

                for i in range(len(total_ball_box)):
                    x0, y0, x1, y1 = total_ball_box[i]

                    ball_x_pos, ball_y_pos = int((x0 + x1)/2), int((y0 +y1)/2)

                    cv2.rectangle(frame_main, (x0, y0), (x1, y1), color, 3)

                    if record == True :
                        cv2.rectangle(frame_record, (x0, y0), (x1, y1), color, 3)

                    #cv2.circle(point_image,(ball_x_pos, ball_y_pos), 4, color, -1)

                    #ball_list.append([ball_x_pos, ball_y_pos])

            ball_cen_left = trans_point(frame_main, ball_box_left)
            ball_cen_right = trans_point(frame_main, ball_box_right)

                

            #print("ball_cen_left = ",ball_cen_left)
            #print("ball_cen_right = ",ball_cen_right)

            #print("KF_flag : ",estimation_ball.kf_flag)

            if len(ball_cen_left) and len(ball_cen_right): #2개의 카메라에서 ball이 검출 되었는가?
                fly_check = estimation_ball.check_ball_flying(ball_cen_left, ball_cen_right)
                if (fly_check) == 1:

                    ball_cand_pos = estimation_ball.get_ball_pos()

                    print("check_ball_fly")

                    if estimation_ball.kf_flag:
                        print("ball_detect_next")


                        pred_ball_pos = estimation_ball.kf.get_predict()
                        ball_pos = get_prior_ball_pos(ball_cand_pos, pred_ball_pos)

                        ball_pos = estimation_ball.ball_vel_check(ball_pos)

                        ball_pos_jrajectory.append(ball_pos)

                        estimation_ball.kf.update(ball_pos[0], ball_pos[1], ball_pos[2], dT)

                        estimation_ball.ball_trajectory.append([ball_pos])

                    else:
                        print("ball_detect_frist")

                        if len(ball_cand_pos) > 1:
                            pass
                            #***************사람 위치와 공 위치 평가 함수******************
                            #임시

                            del_list = []

                            for i in range(len(ball_cand_pos)) : #임시

                                if abs(ball_cand_pos[i][1]) > 13.1 / 2 :

                                    del_list.append(i)

                            ball_cand_pos = np.delete(np.array(ball_cand_pos),del_list,axis = 0).tolist()

                            ball_pos = ball_cand_pos[(9 - abs(np.array(ball_cand_pos)[:,0])).argmin()]

                            ball_pos_jrajectory.append(ball_pos)

                            estimation_ball.kf = Kalman_filiter(ball_pos[0], ball_pos[1], ball_pos[2], dT)
                            estimation_ball.kf_flag = True
                            estimation_ball.ball_trajectory.append([ball_pos])

                        else:
                            ball_pos = ball_cand_pos[0]
                            ball_pos_jrajectory.append(ball_pos)

                            estimation_ball.kf = Kalman_filiter(ball_pos[0], ball_pos[1], ball_pos[2], dT)
                            estimation_ball.kf_flag = True
                            estimation_ball.ball_trajectory.append([ball_pos])

                elif (fly_check) == 3:
                    print("setup_ball_fly")
                    #estimation_ball.reset_ball()

                else : 
                    print("not_detect_fly_ball")

                    if estimation_ball.kf_flag == True:
                        print("ball_predict_next_KF")
                        
                        estimation_ball.kf.predict(dT)

                        ball_pos = estimation_ball.kf.get_predict()

                        ball_pos = estimation_ball.ball_vel_check(ball_pos)

                        ball_pos_jrajectory.append(ball_pos.tolist())


                        estimation_ball.ball_trajectory.append([ball_pos])

                        print("pred_ball_pos = ",ball_pos)

                    else : 
                        print("reset_ALL")
                        estimation_ball.reset_ball()
                        ball_pos_jrajectory.clear()

                    
            else:
                print("not ball_detect")

                if estimation_ball.kf_flag: #칼만 필터가 있는가?
                    print("ball_predict_next_KF")

                    estimation_ball.kf.predict(dT)

                    ball_pos = estimation_ball.kf.get_predict()

                    ball_pos = estimation_ball.ball_vel_check(ball_pos)

                    ball_pos_jrajectory.append(ball_pos.tolist())

                    estimation_ball.ball_trajectory.append([ball_pos])

                    print("pred_ball_pos = ",ball_pos)

                    disappear_cnt += 1

                    if ball_pos[2] < 0 or disappear_cnt > 4 or  ball_pos[0] > 0 :

                        print("reset_ALL")

                        estimation_ball.reset_ball()
                        ball_pos_jrajectory.clear()
                        disappear_cnt = 0


                else:
                    print("reset_ALL")
                    estimation_ball.reset_ball()
                    ball_pos_jrajectory.clear()

            if len(ball_pos) and (ball_pos[0] < - 2.5):
                #print("ball_pos_jrajectory = ",ball_pos_jrajectory)

                ball_landing_point = cal_landing_point(ball_pos_jrajectory, dT)

                draw_point_court(tennis_court_img, ball_pos, padding_x, padding_y)
                draw_point_court(tennis_court_img, [ball_pose_x, ball_pose_y, ball_pose_z], padding_x, padding_y, [0, 0, 255])

                draw_landing_point_court(tennis_court_img, ball_landing_point, padding_x, padding_y)

                #print("ball_pos = ",ball_pos)
                print("real_ball_pos = ", ball_pose_x, ball_pose_y, ball_pose_z)
                print("real_ball_vel = ", ball_vel_x, ball_vel_y, ball_vel_z)

                print("ball_landing_point = ",ball_landing_point)
                array2data.data = ball_landing_point
                pub.publish(array2data)

            t2 = time.time()


            #cv2.imshow('person_tracking_img',person_tracking_img)
            #cv2.imshow('ball_detect_img',ball_detect_img)

            cv2.imshow('tennis_court_img',tennis_court_img)
            cv2.imshow('frame_main',frame_main)

            #cv2.imshow('frame_record',frame_record)

            if record:

                out.write(frame_record)

            dT = t2-t1

            print("FPS : " , 1/(t2-t1))


            key = cv2.waitKey(1)

            if key == ord("c") : 
                tennis_court_img = cv2.imread(path + "/images/tennis_court.png")

                tennis_court_img = cv2.resize(tennis_court_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA) # 1276,600,0

                padding_y = int((810 - tennis_court_img.shape[0]) /2 )
                padding_x = int((1500 - tennis_court_img.shape[1]) /3)

                WHITE = [255,255,255]
                tennis_court_img= cv2.copyMakeBorder(tennis_court_img.copy(),padding_y,padding_y,padding_x,padding_x,cv2.BORDER_CONSTANT,value=WHITE)

                #print(tennis_court_img.shape)

            if key == 27:
                cv2.destroyAllWindows()
                ray.shutdown()
                break

if __name__ == "__main__":

    main()