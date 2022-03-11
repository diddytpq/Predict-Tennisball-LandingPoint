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

ball_trajectory =  [[-1.8481700897216804, 1.6059576129205053, 1.85512028739923], [-0.8162591934204109, 1.828201439369866, 1.827501704135834], [0.05998535156249929, 1.9841348287091722, 1.7686288075179677], [0.8914491653442376, 2.171445725119472, 1.7459164704608874], [1.6503240585327141, 2.3239164536460377, 1.6309971827780632], [2.30226821899414, 2.4862710046197165, 1.560488839191728], [2.988402557373046, 2.6465845512468444, 1.4654796949429123], [2.988402557373046, 2.6199857115358207, 1.4388808552318886], [3.6095144271850588, 2.7986668816663913, 1.3575389408296272], [4.265814971923828, 2.950201918439154, 1.192158745645462], [4.920106601715088, 3.1151797234235556, 1.0719678483959236], [5.591114711761475, 3.277596847583163, 0.9169059954133846], [6.229037475585938, 3.384958679771817, 0.7282149965146716], [6.884402465820313, 3.526617944926409, 0.545650448813162], [7.538687419891358, 3.658038022393329, 0.3624739148789967]]
real_ball_trajectory =  [[-0.9478398860931945, 1.9650424952172736, 1.7617668947368421], [-0.1569811223873707, 2.1291699353626776, 1.7233570350877188], [0.6481859852407228, 2.3040544055096617, 1.667955270175438], [1.3546673051338627, 2.46585199315625, 1.601600778947368], [1.94358338462916, 2.607719371898147, 1.5297620245614039], [2.4510865061671008, 2.731694790813906, 1.4565404596491236], [2.9749606961417494, 2.859669416791464, 1.371078508771931], [3.5479480914265213, 2.9996416639544177, 1.2661140000000017], [4.0390801445277535, 3.119617875808378, 1.1665894210526342], [4.61206753981251, 3.259590122971332, 1.0393299122807056], [5.201426003533974, 3.4035615771960845, 0.8959100175438649], [5.839897672565559, 3.559530652606233, 0.7262059649122872], [6.396513999413608, 3.6955036927073883, 0.5660970421052703], [6.9531303262616575, 3.8314767328085435, 0.39465931929825326], [7.67345733747678, 4.007441843527685, 0.15598193684211306]]

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

draw_point_3D(ball_trajectory, real_ball_trajectory,True)


#낙하지점
"""draw_point_3D(real_ball_trajectory_list, estimation_ball_trajectory_list)

p = Ellipse((landing_point[0], landing_point[1]),  height = circle_radius_y,  width= circle_radius_x, angle = 0, color = "blue", zorder = 21 )
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0.01, zdir="z")"""






ax.view_init(30, 45)

ax.legend()

plt.show()