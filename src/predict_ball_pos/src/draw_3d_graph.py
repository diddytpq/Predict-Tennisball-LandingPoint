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




#real_ball_trajectory_list =  [[-10.39, 3.249, 1.881], [-9.421, 3.249, 2.055], [-8.463, 3.249, 2.203], [-7.435, 3.249, 2.334], [-5.566, 3.249, 2.497], [-4.74, 3.249, 2.538], [-4.035, 3.249, 2.559], [-3.128, 3.249, 2.565], [-2.403, 3.249, 2.554], [-1.639, 3.249, 2.525]]
#estimation_ball_trajectory_list =  [[-10.684, 3.256, 1.729], [-9.629, 3.252, 1.941], [-8.66, 3.244, 2.112], [-7.479, 3.512, 2.187], [-5.53, 3.238, 2.468], [-4.707, 3.236, 2.516], [-4.025, 3.398, 2.524], [-3.075, 3.227, 2.56], [-2.331, 3.233, 2.547], [-1.591, 3.227, 2.521]]


#3
#real_ball_trajectory_list =  [[-10.332, -0.007, 1.862], [-9.363, -0.007, 2.04], [-8.363, -0.007, 2.197], [-7.433, -0.007, 2.318], [-6.453, -0.007, 2.42], [-5.541, -0.007, 2.492], [-4.618, -0.007, 2.541], [-3.723, -0.007, 2.566], [-2.799, -0.007, 2.569], [-0.961, -0.007, 2.501]]
#estimation_ball_trajectory_list =  [[-10.606, -0.002, 1.717], [-9.597, -0.004, 1.912], [-8.404, 0.305, 2.063], [-7.726, -0.01, 2.227], [-6.758, -0.006, 2.34], [-5.801, -0.008, 2.428], [-4.849, -0.009, 2.512], [-3.991, -0.009, 2.533], [-3.009, -0.006, 2.554], [-1.151, 0.047, 2.518]]

#5
#real_ball_trajectory_list =  [[-10.316, -3.72, 1.894], [-9.489, -3.72, 2.044], [-8.668, -3.72, 2.174], [-7.815, -3.72, 2.29], [-6.93, -3.72, 2.389], [-6.093, -3.72, 2.464], [-5.243, -3.72, 2.519], [-4.476, -3.72, 2.552], [-3.678, -3.72, 2.57], [-2.944, -3.72, 2.57], [-2.187, -3.72, 2.555]]
#estimation_ball_trajectory_list =  [[-10.309, -3.711, 1.807], [-9.442, -3.709, 1.983], [-8.639, -3.71, 2.107], [-7.794, -3.715, 2.219], [-6.946, -3.719, 2.326], [-6.102, -3.715, 2.428], [-5.199, -3.712, 2.485], [-4.409, -3.713, 2.531], [-3.63, -3.7, 2.544], [-2.871, -3.708, 2.565], [-2.168, -3.711, 2.551]]


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

estimation_ball_trajectory_list_1 = np.array( [[-10.162, 3.305, 1.839], [-9.203, 3.305, 2.027], [-8.285, 3.299, 2.151], [-7.357, 3.301, 2.285], [-6.046, 3.289, 2.423], [-5.159, 3.293, 2.478], [-4.203, 3.291, 2.534], [-3.294, 3.281, 2.547], [-2.409, 3.284, 2.559]] )

estimation_ball_trajectory_list_2 = np.array( [[-10.981, 2.225, 1.624], [-9.858, 2.17, 1.893], [-8.805, 2.172, 2.089], [-7.811, 2.169, 2.237], [-6.825, 2.172, 2.356], [-5.867, 2.168, 2.432], [-4.924, 2.165, 2.491], [-4.004, 2.16, 2.548], [-3.115, 2.156, 2.565], [-2.259, 2.163, 2.543], [-1.403, 2.157, 2.512], [-0.529, 2.155, 2.435]] )

estimation_ball_trajectory_list_3 = np.array( [[-11.254, 0.036, 1.632], [-10.161, 0.036, 1.838], [-9.108, 0.033, 2.048], [-8.183, 0.028, 2.188], [-7.192, 0.035, 2.297], [-5.934, 0.032, 2.39], [-4.896, 0.031, 2.518], [-3.996, 0.029, 2.546], [-2.979, 0.032, 2.56], [-2.066, 0.028, 2.548], [-1.228, 0.031, 2.512], [-0.547, 0.03, 2.445]] )

estimation_ball_trajectory_list_4 = np.array( [[-11.054, -2.165, 1.669], [-9.95, -2.166, 1.883], [-8.91, -2.166, 2.064], [-7.915, -2.167, 2.216], [-6.984, -2.161, 2.343], [-6.026, -2.163, 2.423], [-5.131, -2.158, 2.485], [-4.157, -2.158, 2.539], [-3.271, -2.163, 2.553], [-2.355, -2.164, 2.553], [-1.484, -2.157, 2.518], [-0.784, -2.157, 2.47]] )

estimation_ball_trajectory_list_5 = np.array( [[-10.098, -3.9, 1.857], [-9.15, -3.896, 2.032], [-8.209, -3.889, 2.178], [-7.254, -3.889, 2.295], [-6.413, -3.89, 2.377], [-5.16, -3.889, 2.497], [-4.201, -3.886, 2.533], [-3.303, -3.888, 2.554], [-2.513, -3.886, 2.548]] )

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


esti_ball_pos_list =  [[0.15927508074515906, -1.1817381200731987, 1.8834861254364959], [1.3657249121932082, -1.369551486183954, 1.8443025341464], [2.5254757114713744, -1.4221383000977288, 1.7330601298387303], [3.522848866614524, -1.5294876888910534, 1.638961401603235], [4.316406306789814, -1.6264330647116099, 1.5351561396272635], [5.058409186654659, -1.6998594369552944, 1.3929592386019545], [5.78441299398749, -1.7985993627702692, 1.2235166903225714], [6.591311948761448, -1.8361467279910584, 1.0809209087808866], [7.230869458489514, -1.9571500744679455, 0.9103855838546808], [7.926988466914298, -2.0328977141149487, 0.68010416133901], [8.48706643228039, -2.1498631240676573, 0.5138334548877382], [9.36193698771153, -2.2268381040446785, 0.254963626417807], [10.230538488078107, -2.294399414310892, 0.2691384895798903], [10.775013929914085, -2.3747067677553524, 0.42800017706231086], [11.534448496606858, -2.4148904535603766, 0.6186332552701095], [12.093424832828342, -2.477345667691554, 0.7474035033762763], [12.850950147314766, -2.368129185469787, 0.9327246470546213]]
real_ball_trajectory =  [[-11.5, -8.4e-05, 1.29998], [-9.906371352281468, -0.1433388437548033, 1.4506891964912279], [-8.386689722106798, -0.28070598575522065, 1.5718252631578946], [-6.7339579848249365, -0.43073052027466796, 1.6768775087719294], [-5.350389302018689, -0.5566830043803485, 1.7422829333333323], [-4.1843643546672675, -0.6629845868254337, 1.7802260877192975], [-2.636821113232064, -0.8040212514452872, 1.8051646035087712], [-1.1075108270151939, -0.9441595289913224, 1.7996509999999997], [0.09689369974035862, -1.0600917564932248, 1.7701613228070174], [1.4917090644393414, -1.202767180204984, 1.6984578807017552], [2.5760385611825303, -1.3205119402071581, 1.605668673684211], [3.557581850630719, -1.4336058928221527, 1.4831271052631583], [4.282849431405132, -1.5190554204178421, 1.367774929824562], [5.107017136830614, -1.6161571563220347, 1.2136629122807037], [5.815801363496529, -1.6996646491996403, 1.0615314771929854], [6.508102236053935, -1.781230107359162, 0.8954449824561446], [7.18391975450283, -1.8608535308006, 0.7166382280701807], [7.942154043494274, -1.9501871278324572, 0.4964159719298306], [8.601488207834628, -2.0278685165558143, 0.28806235789474316], [9.376170088706635, -2.1191212924293525, 0.023389572543037935], [10.169468863583372, -2.2047021894469347, 0.17637912850963944], [10.828635548873617, -2.274339807435977, 0.3671952908753961], [11.516461655263438, -2.347005147946282, 0.5441990081266206], [12.161298630003895, -2.415128904674693, 0.6896334930496439], [12.763146473094988, -2.47871107762121, 0.8074677456444661], [13.393653737285657, -2.5453209730889896, 0.9123715531247563], [13.92385302762781, -2.6013338397323498, 0.9859008185059097], [14.525700870718904, -2.6649160126788667, 1.0531082711007325], [15.098889292710421, -2.725470463104121, 1.1010433688100876], [15.672077714701938, -2.786024913529375, 1.1332984665194428]]

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



#draw_point_3D(real_ball_trajectory_list_1, estimation_ball_trajectory_list_1)

#draw_point_3D(real_ball_trajectory_list_2, estimation_ball_trajectory_list_2)

#draw_point_3D(real_ball_trajectory_list_3, estimation_ball_trajectory_list_3)

#draw_point_3D(real_ball_trajectory_list_4, estimation_ball_trajectory_list_4)

#draw_point_3D(real_ball_trajectory_list_5, estimation_ball_trajectory_list_5,True)

draw_point_3D(real_ball_trajectory,esti_ball_pos_list,True)


#낙하지점
"""draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list)

p = Ellipse((landing_point[0], landing_point[1]),  height = circle_radius_y,  width= circle_radius_x, angle = 0, color = "blue", zorder = 21 )
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0.01, zdir="z")"""






ax.view_init(30, 45)

ax.legend()

plt.show()