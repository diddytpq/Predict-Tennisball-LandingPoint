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

    real_landing_list = real_ball_landing_point_list

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

    cv2.putText(img,str(index),(esti_pix_point_xy[0] + 5,esti_pix_point_xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)


    return img

if __name__ == "__main__" :


    trajectory_list = [[19, 307], [67, 310], [108, 312], [147, 314], [184, 316], [220, 318], [258, 319], [292, 321], [328, 322], [362, 323], [391, 324], [420, 325], [450, 326], [483, 327], [511, 328], [542, 329], [571, 331], [601, 332]]

    real_ball_landing_point_list =  [3.668, -0.731]
    esti_ball_landing_point_list =  [2.719, 1.571, 3.003, -1.839, 2.335, -1.06, 2.12, -1.104, 2.369, -0.663, 2.918, -1.02, 3.253, -1.023, 2.671, -0.984, 3.728, -0.77, 2.512, -0.843, 3.122, -0.657, 2.226, -0.393, 3.067, -1.042, 2.789, -0.736, 3.704, -0.898, 2.922, -0.494, 3.236, -1.034]

    esti_landing_point = np.array(esti_ball_landing_point_list)
    esti_landing_point= esti_landing_point.reshape(-1,2)


    count = 0

    index = 0


    cv2.circle(landing_img,(trajectory_list[0][0],trajectory_list[0][1]), 4, [0, 255, 0], -1)


    print(len(esti_landing_point))

    for i in range(len(esti_landing_point)):
        landing_img = draw_real_landing_point(index, real_ball_landing_point_list, landing_img.copy())

        
        esti_x, esti_y = esti_landing_point[i]

        landing_img = draw_esti_landing_point(i-count, [esti_x, esti_y], landing_img)

        cv2.circle(landing_img,(trajectory_list[i][0],trajectory_list[i][1]), 4, [0, 255, 0], -1)
        cv2.circle(landing_img,(trajectory_list[i+1][0],trajectory_list[i+1][1]), 4, [0, 255, 0], -1)


        cv2.imshow("landing_img",landing_img)

        key = cv2.waitKey(0)

        cv2.imwrite("landing_img_{}.png".format(i), landing_img)

        landing_img = cv2.imread(path + "/images/tennis_court.png")
        landing_img = cv2.resize(landing_img,(0,0), fx=2, fy=2, interpolation = cv2.INTER_AREA)

        if key == ord('s'):

            break