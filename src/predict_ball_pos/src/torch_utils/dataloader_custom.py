import pandas as pd
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
from PIL import Image
import random
import cv2
from network import *
import math
import albumentations as A


HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

TP = TN = FP1 = FP2 = FN = 0


def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
      return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag


def getData(data_path_x, data_path_y, mode):
    if mode == 'train':
        img = pd.read_csv(data_path_x)
        label = pd.read_csv(data_path_y)
        return np.squeeze(img.values), np.squeeze(label.values)

    if mode == 'test':
        img = pd.read_csv(data_path_x)
        label = pd.read_csv(data_path_y)
        return np.squeeze(img.values), np.squeeze(label.values)


class TrackNetLoader(data.Dataset):
    def __init__(self, data_path_x, data_path_y, mode, augmentation):
        self.data_path_x = data_path_x
        self.data_path_y = data_path_y

        self.img_name, self.label_name = getData(self.data_path_x, self.data_path_y, mode)
        self.mode = mode
        self.augmentation = augmentation

        img = Image.open(self.img_name[0][0]).convert('LA')
        w, h = img.size
        self.ratio = h / HEIGHT
        print("> Found %d data..." % (len(self.img_name)))

        if self.augmentation:
            self.transform = A.Compose([
                    A.HorizontalFlip(p=0.3),
                    A.Cutout(num_holes=30, max_h_size = 16, max_w_size = 16, fill_value=0, p = 0.3),
                    #A.RandomContrast(limit=0.8, p=0.5),
                    #A.ShiftScaleRotate(scale_limit=0.50, rotate_limit=45, p=.5),
                    A.Rotate(limit = (-45,45), p = 0.4),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit = 0.2, p=0.3),
                    A.Blur(blur_limit=(3, 3), p =0.2),
                    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.2),
                    #A.ChannelShuffle(p=0.3)
                ],
                additional_targets = {
                    'image1' :'image',
                    'image2' :'image'
                })

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img_path = self.img_name[index]
        label_path = self.label_name[index]
        print(img_path)
        print(label_path)
        img_all = []
        label_all = []
        """for i in range(3):
            x = Image.open(img_path[i]).convert('RGB')
            x = x.resize((WIDTH, HEIGHT))
            
            x = np.asarray(x).transpose(2, 0, 1) / 255.0
            #x = x.resize((WIDTH, HEIGHT, 3))

            img_all.append(x[0])
            img_all.append(x[1])
            img_all.append(x[2])
            #print(x[0].shape)"""

        x_0 = Image.open(img_path[0]).convert('RGB')
        x_1 = Image.open(img_path[1]).convert('RGB')
        x_2 = Image.open(img_path[2]).convert('RGB')

        x_0 = x_0.resize((WIDTH, HEIGHT))
        x_1 = x_1.resize((WIDTH, HEIGHT))
        x_2 = x_2.resize((WIDTH, HEIGHT))

        y = Image.open(label_path)


        if self.augmentation:
            self.transformed = self.transform(image = np.array(x_0), image1= np.array(x_1), image2 = np.array(x_2), mask = np.array(y))

            x_0 = self.transformed['image']
            x_1 = self.transformed['image1']
            x_2 = self.transformed['image2']
            y = self.transformed['mask']

        x_0 = np.asarray(x_0).transpose(2, 0, 1) / 255.0
        x_1 = np.asarray(x_1).transpose(2, 0, 1) / 255.0
        x_2 = np.asarray(x_2).transpose(2, 0, 1) / 255.0

        for i in [x_0, x_1, x_2]:
            img_all.append(i[0])
            img_all.append(i[1])
            img_all.append(i[2])

        y = np.asarray(y) / 255.0
        label_all.append(y)

        img_all = np.asarray(img_all)
        label_all = np.asarray(label_all)
       
        return img_all, label_all


def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    tp = tn = fp1 = fp2 = fn = 0

    while i < n:
        if np.max(y_pred[i]) == 0 and np.max(y_true[i]) == 0:
            tn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) == 0:
            fp2 += 1
        elif np.max(y_pred[i]) == 0 and np.max(y_true[i]) > 0:
            fn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) > 0:
            #h_pred
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(y_pred[i].copy(), connectivity = 8)
            if len(stats): 
                stats = np.delete(stats, 0, axis = 0)
                centroids = np.delete(centroids, 0, axis = 0)

            (cx_pred, cy_pred) = centroids[np.argmax(stats[:,-1])]

            #h_true
            (cnts, _) = cv2.findContours(y_true[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
            dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))

            if dist > tol:
                fp1 += 1
            else:
                tp += 1
        i += 1
    return (tp, tn, fp1, fp2, fn)


def evaluation(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    
    try:
        accuracy_2 = (TP + TN) / (TP + TN + FP1 +  FN)
    except:
        accuracy_2 = 0

        
    try:
        f1_score = 2 * (precision * recall)/(precision + recall)
    except:
        f1_score = 0

    return (accuracy, precision, recall, accuracy_2, f1_score)

def display(TP, TN, FP1, FP2, FN):
    print('======================Evaluate=======================')
    print("Number of true positive:", TP)
    print("Number of true negative:", TN)
    print("Number of false positive FP1:", FP1)
    print("Number of false positive FP2:", FP2)
    print("Number of false negative:", FN)
    (accuracy, precision, recall, accuracy_2, f1_score)= evaluation(TP, TN, FP1, FP2, FN)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("accuracy_2:", accuracy_2)
    print("F1 score:", f1_score)
    print('=====================================================')

if __name__ == '__main__' :
    batchsize = 1

    test_data_path_x = 'data_path_csv/test_input.csv'
    test_data_path_y = 'data_path_csv/test_label.csv'

    train_data = TrackNetLoader(test_data_path_x, test_data_path_y , 'test')
    train_loader = DataLoader(dataset = train_data, batch_size=batchsize, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())


    model = efficientnet_b3()
    model.to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=1, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    checkpoint = torch.load('weights/custom_1.tar')
    model.load_state_dict(checkpoint['state_dict'])

    for batch_idx, (data, label) in enumerate(train_loader):


        img_0, img_1, img_2 = np.array(data[0,0:3,:,:]), np.array(data[0,3:6,:,:]), np.array(data[0,6:,:,:])

        img_0 = (img_0.transpose(1, 2, 0) * 255).astype('uint8')
        img_1 = (img_1.transpose(1, 2, 0) * 255).astype('uint8')
        img_2 = (img_2.transpose(1, 2, 0) * 255).astype('uint8')


        data = data.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            torch.cuda.synchronize()

            y_pred = model(data)

            y_true = np.array(label[0])

            y_pred = (y_pred * 255).cpu().numpy()
            y_pred = y_pred[0].astype('uint8')
            y_pred = (50 < y_pred) * y_pred
                
            torch.cuda.synchronize()

            y_true = (y_true * 255).astype('uint8')


        (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_true, 25)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn

        debug_img = cv2.hconcat([y_pred[0], y_true[0]])
        input_img = cv2.hconcat([img_0, img_1,img_2])

        cv2.imshow("input_img",input_img)
        cv2.imshow("debug_img",debug_img)

        display(TP, TN, FP1, FP2, FN)

        key = cv2.waitKey(1)



        if key == 27 : 
            cv2.destroyAllWindows()
            break
