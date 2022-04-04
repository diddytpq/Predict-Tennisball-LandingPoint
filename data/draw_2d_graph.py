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

from pathlib import Path
import sys

import pickle
import time

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

path = str(FILE.parents[0])

plt.figure(figsize = (10,10))


with open("data/real_.bin","rb") as fr:
    real_data = pickle.load(fr)

with open("data/esti_.bin","rb") as fr:
    esti_data = pickle.load(fr)


#basic

i, j = list(input().split())
check_list = []

real_sim_time, real_ball_trajectory = np.array(real_data[int(i)])[:,0], np.array(real_data[int(i)])[:,1:]

remove_time = np.argmin(real_sim_time)

real_sim_time[remove_time:] += 1e+9
# real_sim_time, real_ball_trajectory = real_ball_trajectory[remove_time:,0], real_ball_trajectory[remove_time:,1:]

sim_time, measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)][0], esti_data[int(i)][int(j)][1], esti_data[int(i)][int(j)][2]

sim_time_list = np.arange(sim_time[-1] + 1e+9, sim_time[-1] + 1e+9 + len(esti_ball_pos_list)*1e+7, 1e+7)

if np.argmin(sim_time) != 0:
    measure_sim_time_list = sim_time[:np.argmin(sim_time)].tolist() + (np.array(sim_time[np.argmin(sim_time):]) + 1e+9).tolist()

else :
    measure_sim_time_list = sim_time +  1e+9

# x axis
plt.subplot(2, 1, 1)
plt.plot(real_sim_time,real_ball_trajectory[:,0],'ro', measure_sim_time_list,measure_data[:,0],'b*',sim_time_list, np.array(esti_ball_pos_list)[:,0],'bo')#, x,y2,'g-')

# # y axis
# plt.subplot(3, 1, 2)
# plt.plot(real_sim_time,real_ball_trajectory[:,1],'ro', measure_sim_time_list,measure_data[:,1],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,1],'bo')#, x,y2,'g-')

# z axis
plt.subplot(2, 1, 2)
plt.plot(real_sim_time,real_ball_trajectory[:,2],'ro', measure_sim_time_list,measure_data[:,2],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,2],'bo')#, x,y2,'g-')
plt.grid()
plt.show()


# select one trajectroy
# i = int(input())

# real_sim_time, real_ball_trajectory = np.array(real_data[int(i)])[:,0], np.array(real_data[int(i)])[:,1:]

# remove_time = np.argmin(real_sim_time)

# real_sim_time[remove_time:] += 1e+9
# # real_sim_time, real_ball_trajectory = real_ball_trajectory[remove_time:,0], real_ball_trajectory[remove_time:,1:]

# for j in range(len(esti_data[int(i)])):
#     sim_time, measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)][0], esti_data[int(i)][int(j)][1], esti_data[int(i)][int(j)][2]
    
#     if np.argmin(np.array(esti_ball_pos_list)[:,2]) == 0:

#         j = j +1
#         break 

# sim_time, measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)][0], esti_data[int(i)][int(j)][1], esti_data[int(i)][int(j)][2]

# sim_time_list = np.arange(sim_time[-1] + 1e+9, sim_time[-1] + 1e+9 + len(esti_ball_pos_list)*1e+7, 1e+7)

# if np.argmin(sim_time) != 0:
#     measure_sim_time_list = sim_time[:np.argmin(sim_time)].tolist() + (np.array(sim_time[np.argmin(sim_time):]) + 1e+9).tolist()

# else :
#     measure_sim_time_list = sim_time +  1e+9

# # x axis
# plt.subplot(2, 1, 1)
# plt.plot(real_sim_time,real_ball_trajectory[:,0],'ro', measure_sim_time_list,measure_data[:,0],'b*',sim_time_list, np.array(esti_ball_pos_list)[:,0],'bo')#, x,y2,'g-')

# # # y axis
# # plt.subplot(3, 1, 2)
# # plt.plot(real_sim_time,real_ball_trajectory[:,1],'ro', measure_sim_time_list,measure_data[:,1],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,1],'bo')#, x,y2,'g-')

# # z axis
# plt.subplot(2, 1, 2)
# plt.plot(real_sim_time,real_ball_trajectory[:,2],'ro', measure_sim_time_list,measure_data[:,2],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,2],'bo')#, x,y2,'g-')
# # plt.plot(real_sim_time,real_ball_trajectory[:,2],'ro', measure_sim_time_list,measure_data[:,2],'b*')#, x,y2,'g-')

# plt.grid()
# plt.show()



"""total_MAE = []

for i in range(1,len(real_data)):
    real_sim_time, real_ball_trajectory = np.array(real_data[int(i)])[:,0], np.array(real_data[int(i)])[:,1:]

    remove_time = np.argmin(real_sim_time)

    real_sim_time[remove_time:] += 1e+9
    # real_sim_time, real_ball_trajectory = real_ball_trajectory[remove_time:,0], real_ball_trajectory[remove_time:,1:]

    for j in range(len(esti_data[int(i)])):
        sim_time, measure_data,  esti_ball_pos_list = esti_data[int(i)][int(j)][0], esti_data[int(i)][int(j)][1], esti_data[int(i)][int(j)][2]
        
        if np.argmin(np.array(esti_ball_pos_list)[:,2]) == 0:

            j = j 
            break 


    sim_time_list = np.arange(sim_time[-1] + 1e+9, sim_time[-1] + 1e+9 + len(esti_ball_pos_list)*1e+7, 1e+7)

    if np.argmin(sim_time) != 0:
        measure_sim_time_list = sim_time[:np.argmin(sim_time)].tolist() + (np.array(sim_time[np.argmin(sim_time):]) + 1e+9).tolist()

    else :
        measure_sim_time_list = sim_time +  1e+9

# x axis
plt.subplot(2, 1, 1)
plt.plot(real_sim_time,real_ball_trajectory[:,0],'ro', measure_sim_time_list,measure_data[:,0],'b*',sim_time_list, np.array(esti_ball_pos_list)[:,0],'bo')#, x,y2,'g-')

# # y axis
# plt.subplot(3, 1, 2)
# plt.plot(real_sim_time,real_ball_trajectory[:,1],'ro', measure_sim_time_list,measure_data[:,1],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,1],'bo')#, x,y2,'g-')

# z axis
plt.subplot(2, 1, 2)
plt.plot(real_sim_time,real_ball_trajectory[:,2],'ro', measure_sim_time_list,measure_data[:,2],'b*', sim_time_list, np.array(esti_ball_pos_list)[:,2],'bo')#, x,y2,'g-')
plt.grid()
plt.show()"""