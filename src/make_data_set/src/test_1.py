import cv2
import numpy as np
import pickle



if __name__ == "__main__":

      with open('list_600.bin', 'rb') as f:
            data = pickle.load(f)

data_point = []
for i in range(len(data)):
    data_point.append(data[i][0])

print(data_point[0][0])