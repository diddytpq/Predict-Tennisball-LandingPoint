from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

import cv2

from pathlib import Path
import sys

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

            ax.plot(real_x, real_y, real_z, 'red', zorder = 100)

        


        for j in range((len(estimation_ball_trajectory_list))):

            esti_x.append(estimation_ball_trajectory_list[j][0])
            esti_y.append(estimation_ball_trajectory_list[j][1])
            esti_z.append(estimation_ball_trajectory_list[j][2])

            ax.plot(esti_x, esti_y, esti_z, '#3336FF', zorder = 100)


        if label == True:
            ax.plot(real_x[-1], real_y[-1], real_z[-1], 'red', zorder = 100, label = 'original trajectory')
            ax.plot(esti_x[-1], esti_y[-1], esti_z[-1], '#3336FF', zorder = 100, label = 'predict trajectory')
#1
#real_ball_trajectory_list =  [[-10.39, 3.249, 1.881], [-9.421, 3.249, 2.055], [-8.463, 3.249, 2.203], [-7.435, 3.249, 2.334], [-5.566, 3.249, 2.497], [-4.74, 3.249, 2.538], [-4.035, 3.249, 2.559], [-3.128, 3.249, 2.565], [-2.403, 3.249, 2.554], [-1.639, 3.249, 2.525]]
#estimation_ball_trajectory_list =  [[-10.684, 3.256, 1.729], [-9.629, 3.252, 1.941], [-8.66, 3.244, 2.112], [-7.479, 3.512, 2.187], [-5.53, 3.238, 2.468], [-4.707, 3.236, 2.516], [-4.025, 3.398, 2.524], [-3.075, 3.227, 2.56], [-2.331, 3.233, 2.547], [-1.591, 3.227, 2.521]]


#3
#real_ball_trajectory_list =  [[-10.332, -0.007, 1.862], [-9.363, -0.007, 2.04], [-8.363, -0.007, 2.197], [-7.433, -0.007, 2.318], [-6.453, -0.007, 2.42], [-5.541, -0.007, 2.492], [-4.618, -0.007, 2.541], [-3.723, -0.007, 2.566], [-2.799, -0.007, 2.569], [-0.961, -0.007, 2.501]]
#estimation_ball_trajectory_list =  [[-10.606, -0.002, 1.717], [-9.597, -0.004, 1.912], [-8.404, 0.305, 2.063], [-7.726, -0.01, 2.227], [-6.758, -0.006, 2.34], [-5.801, -0.008, 2.428], [-4.849, -0.009, 2.512], [-3.991, -0.009, 2.533], [-3.009, -0.006, 2.554], [-1.151, 0.047, 2.518]]

#5
#real_ball_trajectory_list =  [[-10.316, -3.72, 1.894], [-9.489, -3.72, 2.044], [-8.668, -3.72, 2.174], [-7.815, -3.72, 2.29], [-6.93, -3.72, 2.389], [-6.093, -3.72, 2.464], [-5.243, -3.72, 2.519], [-4.476, -3.72, 2.552], [-3.678, -3.72, 2.57], [-2.944, -3.72, 2.57], [-2.187, -3.72, 2.555]]
#estimation_ball_trajectory_list =  [[-10.309, -3.711, 1.807], [-9.442, -3.709, 1.983], [-8.639, -3.71, 2.107], [-7.794, -3.715, 2.219], [-6.946, -3.719, 2.326], [-6.102, -3.715, 2.428], [-5.199, -3.712, 2.485], [-4.409, -3.713, 2.531], [-3.63, -3.7, 2.544], [-2.871, -3.708, 2.565], [-2.168, -3.711, 2.551]]


real_ball_trajectory_list_1 = np.array( [[-10.109, 3.561, 1.942], [-9.124, 3.561, 2.111], [-8.248, 3.561, 2.238], [-7.262, 3.561, 2.357], [-6.227, 3.561, 2.453], [-5.339, 3.561, 2.512], [-4.287, 3.561, 2.554], [-3.378, 3.561, 2.566], [-2.446, 3.561, 2.554]] )
estimation_ball_trajectory_list_1 = np.array( [[-10.033, 3.565, 1.878], [-9.044, 3.573, 2.039], [-8.119, 3.56, 2.187], [-7.203, 3.558, 2.307], [-6.238, 3.549, 2.403], [-5.418, 3.548, 2.462], [-4.159, 3.543, 2.531], [-3.276, 3.544, 2.548], [-2.332, 3.54, 2.553]] )

real_ball_trajectory_list_2 = np.array( [[-11.148, 2.095, 1.733], [-10.051, 2.095, 1.951], [-9.066, 2.095, 2.118], [-7.993, 2.095, 2.271], [-6.99, 2.095, 2.386], [-5.919, 2.095, 2.478], [-5.014, 2.095, 2.531], [-4.098, 2.095, 2.562], [-3.154, 2.095, 2.569], [-2.208, 2.095, 2.551], [-1.324, 2.095, 2.511], [-0.411, 2.095, 2.442]] )
estimation_ball_trajectory_list_2 = np.array( [[-11.059, 2.104, 1.654], [-9.958, 2.099, 1.885], [-8.949, 2.091, 2.067], [-7.911, 2.093, 2.216], [-6.883, 2.096, 2.351], [-5.867, 2.086, 2.433], [-4.941, 2.093, 2.5], [-3.984, 2.082, 2.539], [-3.058, 2.082, 2.563], [-2.185, 2.084, 2.544], [-1.25, 2.079, 2.51], [-0.55, 2.079, 2.446]] )

real_ball_trajectory_list_3 = np.array( [[-11.147, 0.039, 1.75], [-10.092, 0.039, 1.957], [-9.127, 0.039, 2.12], [-8.113, 0.039, 2.264], [-7.051, 0.039, 2.386], [-6.059, 0.039, 2.471], [-5.095, 0.039, 2.528], [-4.161, 0.039, 2.559], [-3.141, 0.039, 2.565], [-2.218, 0.039, 2.545], [-1.285, 0.039, 2.498], [-0.189, 0.039, 2.407]] )
estimation_ball_trajectory_list_3 = np.array( [[-11.189, 0.038, 1.63], [-10.062, 0.052, 1.886], [-8.989, 0.039, 2.073], [-7.992, 0.042, 2.219], [-6.947, 0.041, 2.331], [-5.967, 0.037, 2.433], [-4.963, 0.039, 2.502], [-4.027, 0.036, 2.55], [-3.099, 0.041, 2.55], [-2.243, 0.038, 2.542], [-1.389, 0.033, 2.515], [-0.461, 0.031, 2.421]] )

real_ball_trajectory_list_4 = np.array( [[-11.33, -2.122, 1.711], [-10.151, -2.122, 1.946], [-8.986, -2.122, 2.142], [-8.073, -2.122, 2.27], [-7.109, -2.122, 2.38], [-6.096, -2.122, 2.468], [-4.979, -2.122, 2.533], [-4.064, -2.122, 2.56], [-3.177, -2.122, 2.565], [-2.217, -2.122, 2.545], [-1.283, -2.122, 2.498], [-0.411, -2.122, 2.429]] )
estimation_ball_trajectory_list_4 = np.array( [[-11.764, -2.109, 1.509], [-10.059, -2.123, 1.889], [-8.987, -2.123, 2.07], [-7.994, -2.113, 2.223], [-7.008, -2.114, 2.346], [-5.969, -2.113, 2.444], [-4.964, -2.116, 2.488], [-4.014, -2.117, 2.541], [-3.054, -2.113, 2.555], [-2.125, -2.114, 2.538], [-1.257, -2.11, 2.5], [-0.558, -2.108, 2.437]] )

real_ball_trajectory_list_5 = np.array( [[-10.286, -3.739, 1.928], [-9.3, -3.739, 2.099], [-8.324, -3.739, 2.243], [-7.26, -3.739, 2.37], [-6.324, -3.739, 2.456], [-5.416, -3.739, 2.517], [-4.478, -3.739, 2.555], [-3.437, -3.739, 2.57], [-2.489, -3.739, 2.556], [-1.709, -3.739, 2.526]] )
estimation_ball_trajectory_list_5 = np.array( [[-10.172, -3.743, 1.863], [-9.165, -3.736, 2.056], [-8.17, -3.735, 2.198], [-7.176, -3.734, 2.31], [-6.263, -3.732, 2.419], [-5.312, -3.73, 2.491], [-4.407, -3.728, 2.53], [-3.446, -3.723, 2.552], [-2.516, -3.73, 2.553], [-1.651, -3.726, 2.491]] )




plt.rcParams["figure.autolayout"] = True
fig_3d = plt.figure(figsize = (6,4))
ax = fig_3d.add_subplot(projection='3d')

ax.set_xlim(-11.885, 11.885)
ax.set_ylim(-5.485, 5.485)
ax.set_zlim(0,5)

ax.set_box_aspect((2, 1, 0.5))
img = plt.imread(path + "/images/tennis_court_2.png")

print(img.shape)


stepX, stepY = 24 / img.shape[1], 12 / img.shape[0]

X1 = np.arange(-12, 12, stepX)
Y1 = np.arange(-6, 6, stepY)
#Y1 = np.arange(-8.4, 8.4, stepY)

print(len(X1), len(Y1))

X1, Y1 = np.meshgrid(X1, Y1)

print(X1.shape)
print(stepX,stepY)

ax.plot_surface(X1, Y1, np.zeros(X1.shape),facecolors=img, zorder = 0)

draw_point_3D(real_ball_trajectory_list_1, estimation_ball_trajectory_list_1)

draw_point_3D(real_ball_trajectory_list_2, estimation_ball_trajectory_list_2)

draw_point_3D(real_ball_trajectory_list_3, estimation_ball_trajectory_list_3)

draw_point_3D(real_ball_trajectory_list_4, estimation_ball_trajectory_list_4)

draw_point_3D(real_ball_trajectory_list_5, estimation_ball_trajectory_list_5,True)



ax.legend()

plt.show()