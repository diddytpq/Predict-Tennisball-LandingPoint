from subprocess import check_call
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, PathPatch
from matplotlib.transforms import Affine2D
import numpy as np
from matplotlib.cbook import get_sample_data
import mpl_toolkits.mplot3d.art3d as art3d

import cv2

from pathlib import Path
import sys

import pickle
import time

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])


def draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list, label = False):
        
        real_x = []
        real_y = []
        real_z = []
        esti_x = []
        esti_y = []
        esti_z = []



        for i in range((len(real_ball_trajectory_list))):

            real_x.append(real_ball_trajectory_list[i][0])
            real_y.append(real_ball_trajectory_list[i][1])
            real_z.append(real_ball_trajectory_list[i][2])

            ax.plot(real_x, real_y, real_z, c= 'red', zorder = 100)

        ax.scatter(real_x[0], real_y[0], real_z[0],s = 180, c='#FF3333', zorder = 101, marker = '*')
        


        for j in range((len(estimation_ball_trajectory_list))):

            esti_x.append(estimation_ball_trajectory_list[j][0])
            esti_y.append(estimation_ball_trajectory_list[j][1])
            esti_z.append(estimation_ball_trajectory_list[j][2])

            ax.plot(esti_x, esti_y, esti_z, '#3336FF', zorder = 100)

        ax.scatter(esti_x[0], esti_y[0], esti_z[0],s = 180, c='#3336FF', zorder = 101, marker = '*')


        if label == True:
            ax.plot(real_x[-1], real_y[-1], real_z[-1], 'red', zorder = 100, label = 'Actual trajectory')
            ax.plot(esti_x[-1], esti_y[-1], esti_z[-1], '#3336FF', zorder = 100, label = 'Predict trajectory')

def draw_trajectory_3D(real_ball_trajectory_list, estimation_ball_trajectory_list, prediect_trajectory_list, label = False):
        
        real_x = []
        real_y = []
        real_z = []
        esti_x = []
        esti_y = []
        esti_z = []
        predict_x = []
        predict_y = []
        predict_z = []




        for i in range((len(real_ball_trajectory_list))):

            real_x.append(real_ball_trajectory_list[i][0])
            real_y.append(real_ball_trajectory_list[i][1])
            real_z.append(real_ball_trajectory_list[i][2])

            ax.plot(real_x, real_y, real_z, c= 'red', zorder = 100)

        ax.scatter(real_x[0], real_y[0], real_z[0],s = 180, c='#FF3333', zorder = 101, marker = '*')

        for j in range((len(estimation_ball_trajectory_list))):

            esti_x.append(estimation_ball_trajectory_list[j][0])
            esti_y.append(estimation_ball_trajectory_list[j][1])
            esti_z.append(estimation_ball_trajectory_list[j][2])

            ax.plot(esti_x, esti_y, esti_z, '#3336FF', zorder = 100)

        ax.scatter(esti_x[0], esti_y[0], esti_z[0],s = 180, c='#3336FF', zorder = 99, marker = '*')
        


        for j in range((len(prediect_trajectory_list))):

            predict_x.append(prediect_trajectory_list[j][0])
            predict_y.append(prediect_trajectory_list[j][1])
            predict_z.append(prediect_trajectory_list[j][2])

            ax.plot(predict_x, predict_y, predict_z, '#3336FF',linestyle='dotted', zorder = 100)

        ax.scatter(predict_x[0], predict_y[0], predict_z[0],s = 180, c='#3336FF', zorder = 99, marker = '*')

        if label == True:
            ax.plot(real_x[-1], real_y[-1], real_z[-1], 'red', zorder = 100, label = 'Actual trajectory')
            ax.plot(esti_x[-1], esti_y[-1], esti_z[-1], '#3336FF', zorder = 100, label = 'Predict trajectory')




#궤적
"""with open('1.bin', 'rb') as f:
    real_ball_trajectory_list_1 = pickle.load(f)

with open('2.bin', 'rb') as f:
    real_ball_trajectory_list_2 = pickle.load(f)

with open('3.bin', 'rb') as f:
    real_ball_trajectory_list_3 = pickle.load(f)

with open('4.bin', 'rb') as f:
    real_ball_trajectory_list_4 = pickle.load(f)

with open('5.bin', 'rb') as f:
    real_ball_trajectory_list_5 = pickle.load(f) 

real_ball_trajectory_list_1 = real_ball_trajectory_list_1[:150]
real_ball_trajectory_list_2 = real_ball_trajectory_list_2[:150]
#real_ball_trajectory_list_3 = real_ball_trajectory_list_3[:150]
real_ball_trajectory_list_4 = real_ball_trajectory_list_4[:150]
real_ball_trajectory_list_5 = real_ball_trajectory_list_5[:150]"""
"""with open('5.bin', 'rb') as f:
    real_ball_trajectory_list_5 = pickle.load(f) 
estimation_ball_trajectory_list_5 = np.array( [[-10.098, -3.9, 1.857], [-9.15, -3.896, 2.032], [-8.209, -3.889, 2.178], [-7.254, -3.889, 2.295], [-6.413, -3.89, 2.377], [-5.16, -3.889, 2.497], [-4.201, -3.886, 2.533], [-3.303, -3.888, 2.554], [-2.513, -3.886, 2.548]] )
"""
#낙후지점
"""with open('real_ball_traj.bin', 'rb') as f:
    real_ball_trajectory_list = pickle.load(f)

estimation_ball_trajectory_list = np.array( [[-11.487, 2.185, 2.123], [-10.365, 2.185, 2.123], [-9.521, 2.126, 2.323], [-8.722, 2.087, 2.426], [-7.922, 2.064, 2.529], [-7.121, 2.037, 2.601], [-6.383, 2.011, 2.63], [-5.64, 1.954, 2.597], [-4.937, 1.974, 2.608], [-4.267, 1.908, 2.563], [-2.984, 1.913, 2.404], [-2.348, 1.899, 2.291], [-1.664, 1.88, 2.137], [-1.074, 1.861, 1.992], [-0.361, 1.84, 1.809]] )

landing_point = [3.82, 1.692, 0.0]

circle_radius_x = 2       # 원의 반지름
circle_radius_y = 0.5       # 원의 반지름"""



plt.rcParams["figure.autolayout"] = True
fig_3d = plt.figure(figsize=(8,8),dpi=100)
ax = fig_3d.add_subplot(projection='3d')

ax.set_xlim(-12, 12)
ax.set_ylim(-6, 6)
ax.set_zlim(0,5)
ax.set_box_aspect((2, 1, 0.5))

img = cv2.imread(path + "/images/tennis_court_2.png")
img = cv2.resize(img, dsize=(1000,400), interpolation=cv2.INTER_LINEAR)

# cv2.imshow("img",img)
# cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
img=cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img[:,:,3] = float(1)

stepX, stepY = 24 / img.shape[1], 12 / img.shape[0]

X1 = np.arange(-12, 12, stepX)
Y1 = np.arange(-6, 6, stepY)
#Y1 = np.arange(-8.4, 8.4, stepY)


X1, Y1 = np.meshgrid(X1, Y1)

ax.plot_surface(X1, Y1, np.ones(X1.shape) * -0.01,rstride=8, cstride=8, facecolors=img, zorder = 20)



#낙하지점
"""draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list)

p = Ellipse((landing_point[0], landing_point[1]),  height = circle_radius_y,  width= circle_radius_x, angle = 0, color = "blue", zorder = 21 )
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0.01, zdir="z")"""

with open("data/real_.bin","rb") as fr:
    real_data = pickle.load(fr)

with open("data/esti_.bin","rb") as fr:
    esti_data = pickle.load(fr)

"""i, j = list(input().split())
check_list = []

real_ball_trajectory = np.array(real_data[int(i)][0])

measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)]



draw_trajectory_3D(real_ball_trajectory,measure_data,esti_ball_pos_list,False)
"""

check_list = []

num, j = list(input().split())

num, j = int(num), int(j)

# num = int(input())
# j = 4

# for i in range(0, len(esti_data)):
for i in [num, -1]:
# for i in range(num, num + 10):

    print(i)

    if i == -1 :
        break

    real_sim_time, real_ball_trajectory = np.array(real_data[i])[:,0], np.array(real_data[i])[:,1:]

    # real_landing_point = real_ball_trajectory[np.argmin(real_ball_trajectory[:,2], axis = 0):]

    if len(esti_data[int(i)]) < j:
        continue

    sim_time, measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)][0], esti_data[int(i)][int(j)][1], esti_data[int(i)][int(j)][2]
    # measure_data,  esti_ball_pos_list = esti_data[int(i)][len(esti_data[int(i)])-1]


    # if measure_data[0][2] < 0.1 :
    #     continue

    # if np.min(np.diff(measure_data, axis = 0)[:,0]) < 0:
    #     continue

    # else:
    #     check_list.append(i)
    #     draw_trajectory_3D(real_ball_trajectory,measure_data,esti_ball_pos_list,False)
    draw_trajectory_3D(real_ball_trajectory,measure_data,esti_ball_pos_list,False)

print(check_list)

ax.view_init(10, 74)

ax.legend()

plt.show()