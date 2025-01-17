#! /home/drcl_yang/anaconda3/envs/py36/bin/python

from pathlib import Path
import sys
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add code to path

import numpy as np
from kalman_utils.filter.kalman import Kalman3D




class Kalman_filiter():

    def __init__(self, x_init, y_init, z_init, dT):
        self.KF = Kalman3D(drg=0.9, dbg=0)

        self.dT = dT

        self.pred = self.KF.init(np.float32([x_init, y_init, z_init]))

    def update(self, x, y, z, dT):

        self.dT = dT
        self.pred = self.KF.track(np.float32([x, y, z]), self.dT )

    def predict(self, dT):

        self.pred = self.KF.predict(dT )


    def get_predict(self):
        return self.pred
