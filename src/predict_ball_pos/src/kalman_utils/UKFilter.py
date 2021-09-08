import cv2
import numpy as np
import os
import math
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise


def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    

    #[x, x', y, y', z, z']

    F = np.array([[1, dt, 0, 0, 0, 0],    #x
                    [0, 1, 0, 0, 0, 0],   #x'
                    [0, 0, 1, dt, 0, 0],  #y
                    [0, 0, 0, 1, 0, 0],   #y'
                    [0, 0, 0, 0, 1, dt],  #z
                    [0, 0, 0, 0, 0, 1]])  #z'
    
    B = np.array([0, 0, 0, 0, (dt**2)/2, dt])   #[x, x', y, y', z, z']

    u = np.array([[-9.8]]) # g
    
    return np.dot(F, x) + B * u
    

def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos, z_pos]

    return x[[0, 2, 4]]

class UK_filter():

    def __init__(self, dt, x_std_meas, y_std_meas, z_std_meas, init_x, init_y, init_z):  
    
        self.init_x = init_x 
        self.init_y = init_y
        self.init_z = init_z
        self.dt = dt
        self.z_std = 0.1

        self.sigmas_points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=1)
        self.f = UnscentedKalmanFilter(dim_x=6, dim_z=6, dt=self.dt, fx=fx, hx=hx, points=self.sigmas_points)

        self.f.x = np.array([self.init_x, 0, self.init_y, 0, self.init_z, 0]) 

        self.f.P = np.eye(6)

        self.f.Q = Q_discrete_white_noise(2, dt = self.dt, var = 0.01**2, block_size = 3)
        
        self.f.R = np.array([[x_std_meas**2,0,0],
                            [0, y_std_meas**2,0],
                            [0, 0, y_std_meas**2]])

        self.f.predict()



class Trajectory_ukf:
    def __init__(self, maxDisappeared = 10):

        self.nextObjectID = 0
        self.point_dict = OrderedDict()
        self.disappeared_dict = OrderedDict()
        self.kf_dict = OrderedDict()
        self.kf_pred_dict = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):

        self.point_dict[self.nextObjectID] = [centroid]
        self.disappeared_dict[self.nextObjectID] = 0
        self.kf_dict[self.nextObjectID] = UK_filter(dt = 0.03, 
                                                    x_std_meas = 0.01, 
                                                    y_std_meas = 0.01,
                                                    z_std_meas = 0.01,
                                                    init_x = centroid[0],
                                                    init_y = centroid[1],
                                                    init_z = centroid[2])

        self.kf_pred_dict[self.nextObjectID] = centroid
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.point_dict[objectID]
        del self.disappeared_dict[objectID]
        del self.kf_dict[objectID]
        del self.kf_pred_dict[objectID]

    def update(self, next_centroid_list):
        
        if len(next_centroid_list) == 0:
       
            for ID in list(self.disappeared_dict.keys()):
                self.disappeared_dict[ID] += 1

                self.kf_dict[ID].f.predict()
                
                pred_point = self.kf_dict[ID].f.x

                x, y, z = pred_point[0], pred_point[2], pred_point[4]
                self.kf_pred_dict[ID] = [x, y, z]
                
                if self.disappeared_dict[ID] >= self.maxDisappeared:
                    self.deregister(ID)

            return self.point_dict
        
        if len(self.point_dict) == 0:
            self.register(next_centroid_list)

        else:
            objectIDs = list(self.point_dict.keys())     
            self.kf_predict_list = list()
            
            for ID in objectIDs:
                
                pred_point = self.kf_dict[ID].f.x

                x, y, z = pred_point[0], pred_point[2], pred_point[4]
                self.kf_pred_dict[ID] = [x, y, z]
                
                self.kf_predict_list.append([x, y, z])

            #distan = dist.cdist(np.array(self.kf_predict_list), next_centroid_list)
            
            #ID_list, indexes = linear_sum_assignment(distan)
            
            #next_ball_point = next_centroid_list[indexes[0]]

            next_ball_point = next_centroid_list
            
            self.point_dict[objectIDs[0]].append(next_ball_point)
            self.kf_dict[objectIDs[0]].f.update(next_ball_point)
            self.kf_dict[objectIDs[0]].f.predict()

        return self.point_dict
