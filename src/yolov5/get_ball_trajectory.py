#!/usr/bin/env python

import numpy as np
import time
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox



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

weights = 'last.pt'
img_size = 640

model = attempt_load(weights, map_location=device)  # load FP32 model


stride = int(model.stride.max())  # model stride
imgsz = check_img_size(img_size, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names


point_image = np.zeros([640,640,3], np.uint8) + 255
trajectroy_image = point_image

ball_trajectory_list = []
empty_list = []

trajectroy_cnt = 0

f = open("ball_train_data.txt",'w')

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()
        rospy.init_node('Image_converter', anonymous=True)

        self.image_left_0 = rospy.Subscriber("/camera_left_0/image_raw",Image,self.callback_left_0)
        #self.image_left_1 = rospy.Subscriber("/camera_left_1/image_raw",Image,self.callback_left_1)
        self.image_right_0 = rospy.Subscriber("/camera_right_0/image_raw",Image,self.callback_right_0)
        self.image_right_1 = rospy.Subscriber("/camera_right_1/image_raw",Image,self.callback_right_1)


    def callback_left_0(self,data):
        try:
            self.left_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_left_1(self,data):
        try:
            self.left_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

    def callback_right_0(self,data):
        try:
            self.right_data_0 = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)

    def callback_right_1(self,data):
        try:
            self.right_data_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")


        except CvBridgeError as e:
            print(e)



        (rows,cols,channels) = self.left_data_0.shape

        self.ball_box = []

        global trajectroy_cnt 

        if cols > 60 and rows > 60 :
            t1 = time.time()

            """self.left_data_0 = cv2.resize(self.left_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.left_data_1 = cv2.resize(self.left_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.right_data_0 = cv2.resize(self.right_data_0,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            self.right_data_1 = cv2.resize(self.right_data_1,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)"""

            self.left_frame = cv2.vconcat([self.left_data_0,self.right_data_0])
            #self.right_frame = cv2.vconcat([self.right_data_0,self.right_data_1])

            image_ori = self.left_frame 

            self.image_robot_tracking = self.robot_tracking(image_ori.copy())
            self.image_ball_tracking = self.ball_tracking(image_ori.copy())  
                   
            if self.ball_cand_box:
                self.check_iou(self.robot_box, self.ball_cand_box)

            self.get_ball_centroid()            


            if (max(self.ball_centroid_list) == [0,0] or max(self.ball_centroid_list) > [640, 640]) and len(ball_trajectory_list) == 0:
                pass

            elif (max(self.ball_centroid_list) == [0,0] or max(self.ball_centroid_list) > [640, 640]) :
                ball_trajectory_list.clear()
                f.write("\n")
                trajectroy_cnt += 1
                print(trajectroy_cnt)


            else:
            #elif max(self.ball_centroid_list) > [0,0]:
                ball_trajectory = (self.ball_centroid_list[0] + self.ball_centroid_list[1])
                ball_trajectory_list.append(ball_trajectory)

                save_data = self.make_train_data(ball_trajectory_list)

                print(save_data)
                f.write(str(save_data) + "\n")
                

          

            if self.ball_box:
                for i in range(len(self.ball_box)):
                    x0, y0, x1, y1 = self.ball_box[i]
                    cv2.rectangle(image_ori, (x0, y0), (x1, y1), (255,0,0), 3)
                    cv2.circle(point_image,(int((x0 + x1)/2), int((y0 +y1)/2)), 4, (0,0,255), -1)


            robot_tracking_img = cv2.hconcat([self.image_robot_tracking[:320,:640,:],self.image_robot_tracking[320:,:640,:]])
            ball_detect_img = cv2.hconcat([image_ori[:320,:640,:],image_ori[320:,:640,:]])
            trajectroy_image = cv2.hconcat([point_image[:320,:640,:],point_image[320:,:640,:]])


            t2 = time.time()

            #print(ball_trajectory_list)
            #print(ball_trajectory_list[-1][0])


            #cv2.imshow("left_frame", self.left_frame)
            #cv2.imshow("right_frame", self.right_frame)

            #cv2.imshow("robot_tracking_img", robot_tracking_img)
            #cv2.imshow("image_ball_tracking", self.image_ball_tracking)
            cv2.imshow("ball_detect_img", ball_detect_img)
            #cv2.imshow("trajectroy_image", trajectroy_image)


            #cv2.imshow("fgmask_1", self.fgmask_1)
            #cv2.imshow("fgmask_erode", self.fgmask_erode)
            #cv2.imshow("fgmask_dila", self.fgmask_dila)


            #print(1/(t2-t1))
            
            key = cv2.waitKey(1)


            if key == 27 : 
                cv2.destroyAllWindows()

                return 0

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
                cv2.rectangle(self.image_robot_tracking, (x, y), (x + w, y + h), (255,0,0), 3)

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

    def get_ball_centroid(self):
        self.ball_centroid_list = [[0, 0],[0, 0]]
        for x0, x1, y0, y1 in self.ball_box:
            c_x = (x0 + x1) / 2
            c_y = (y0 + y1) / 2

            if c_y > 320:
                self.ball_centroid_list[1] = [c_x,c_y]
            
            else:
                self.ball_centroid_list[0] = [c_x,c_y]


        """if len(self.ball_box) == 1:
            if self.ball_box
                self.ball_box.append([0,0])"""

    def make_train_data(self, data):
        return data + [[-1, -1, -1, -1]] * (20 - len(data))


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