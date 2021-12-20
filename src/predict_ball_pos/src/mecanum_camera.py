#!/usr/bin/env python

from pathlib import Path
import sys

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])

import numpy as np
import time
import roslib
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

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

roslib.load_manifest('ball_trajectory')

fgbg = cv2.createBackgroundSubtractorMOG2(100, 16, False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dilation_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
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

weights = path + '/weights/yolov5s.pt'
img_size = 640

model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(img_size, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

record = False

class Image_converter:

    def __init__(self):
        
        self.bridge = CvBridge()

        rospy.Subscriber("/mecanum_camera_ir/depth/image_raw",Image,self.callback_depth)
        rospy.Subscriber("/mecanum_camera_ir/mecanum_camera_ir/color/image_raw",Image,self.callback_camera)

        rospy.Subscriber("/camera_left_0_ir/camera_left_0/color/image_raw",Image,self.main)



        self.frame_recode = np.zeros([360,1280,3], np.uint8)



        if record == True:
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            self.out_0 = cv2.VideoWriter("train_video.mp4", self.codec, 35, (1280,640))
            #self.out_1 = cv2.VideoWriter("right.mp4", self.codec, 60, (640,640))

    def callback_camera(self, data):
        try:
            self.t0 = time.time()
            self.camera_data = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)
     
    def callback_depth(self, data):
        try:
            self.camera_depth_data = self.bridge.imgmsg_to_cv2(data, "passthrough")
            self.camera_depth = np.uint8(self.camera_depth_data) 


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

    def main(self, data):
        (rows,cols,channels) = self.camera_data.shape

        if cols > 60 and rows > 60 :
            t1 = time.time()

            #self.camera_data = cv2.resize(self.camera_data,(0,0),fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

            self.camera_depth = cv2.applyColorMap(self.camera_depth, cv2.COLORMAP_JET)

            #self.main_frame = cv2.hconcat([self.camera_data,self.camera_depth])

            ball_image = self.ball_tracking(self.camera_data.copy()) 

            robot_detect_img = self.robot_tracking(self.camera_data.copy()) #get robot bbox

            #self.frame_recode = self.main_frame

            t2 = time.time()

            self.main_frame = cv2.hconcat([self.camera_data, robot_detect_img, ball_image])

            cv2.imshow("main_frame", self.main_frame)
            #cv2.imshow("camera_frame", self.camera_data)
            #cv2.imshow("fgmask_dila", self.fgmask_dila)

            #cv2.imshow("ball_image", ball_image)

            print("FPS : ",1 / (t2 - t1))

            if record == True:
                self.out_0.write(self.main_frame)
                #self.out_1.write(self.right_frame)

            #print((t2-t1))
            key = cv2.waitKey(30)

            if key == 27 : 
                cv2.destroyAllWindows()
                return 0

            if key == ord("s"):
                cv2.imwrite("chess ({}).jpg".format(self.t1), self.left_data_0)





def main(args):

    ic = Image_converter()
    rospy.init_node('Image_converter', anonymous=True)
    
    try:
        rospy.spin()



    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
