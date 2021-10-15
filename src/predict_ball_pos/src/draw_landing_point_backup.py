import rospy
import sys
from pathlib import Path
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
import tf.transformations as tft
import numpy as np
import math
import roslib
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import String, Float64, Float64MultiArray
from nav_msgs.msg import Odometry

import cv2

import pickle

import time

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])


roslib.load_manifest('mecanum_robot_gazebo')

landing_img = cv2.imread(path + "/images/tennis_court.png")
landing_img = cv2.resize(landing_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)

y_pix_length, x_pix_length = landing_img.shape[0], landing_img.shape[1]

x_meter2pix = 23.77 / x_pix_length
y_meter2pix = 10.97 / y_pix_length

def draw_real_landing_point(data_index, landing_point, img):

    real_pix_point_list = []

    real_landing_list = landing_point[data_index]

    real_pix_point_list.append(int(np.round((11.885 + real_landing_list[0]) / x_meter2pix)))
    real_pix_point_list.append(int(np.round((5.485 - real_landing_list[1]) / y_meter2pix)))

    real_pix_point_xy = real_pix_point_list[0:2]

    cv2.circle(img,real_pix_point_xy, 4, [0, 0, 255], -1)

    return img

def draw_esti_landing_point(index, esti_point_list, img):

    predict_pix_point_list = []

    predict_pix_point_list.append(int(np.round((11.885 + esti_point_list[0]) / x_meter2pix)))
    predict_pix_point_list.append(int(np.round((5.485 - esti_point_list[1]) / y_meter2pix)))

    esti_pix_point_xy = predict_pix_point_list[0:2]

    cv2.circle(img,esti_pix_point_xy, 4, [0, 255, 0], -1)

    cv2.putText(img,str(index),(esti_pix_point_xy[0] + 5,esti_pix_point_xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


    return img

if __name__ == "__main__" :

    with open('실험 데이터/관측시작부터낙하지점/real_4.bin', 'rb') as f:
        real_landing_point = pickle.load(f)

    with open('실험 데이터/관측시작부터낙하지점/esti_4.bin', 'rb') as f:
        esti_landing_point = pickle.load(f)

    max_list = []
    for i in range(len(esti_landing_point)):
        max_list.append(len(esti_landing_point[i]))
        
    esti_landing_point = sequence.pad_sequences(esti_landing_point,padding='pre', dtype = 'float32')
    esti_landing_point= esti_landing_point.reshape(100,-1,2)

    print(real_landing_point.shape, esti_landing_point.shape)


    count = 0

    index = 16

    landing_img = draw_real_landing_point(index, real_landing_point, landing_img.copy())

    print(esti_landing_point[index])

    for i in range(17):
        
        esti_list  = esti_landing_point[index]
        
        esti_x, esti_y = esti_list[i]

        if esti_x == 0 :
            count += 1
            continue

        print(i)
        landing_img = draw_esti_landing_point(i-count, esti_list[i], landing_img)



        cv2.imshow("landing_img",landing_img)

        key = cv2.waitKey(0)

        if key == ord('s'):

            break