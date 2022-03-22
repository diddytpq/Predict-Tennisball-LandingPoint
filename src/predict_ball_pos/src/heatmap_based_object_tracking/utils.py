import numpy as np
import cv2
import torch


def find_ball_v2(pred_image, image_ori, ratio_w, ratio_h):

    if np.amax(pred_image) <= 0: #no ball
        return image_ori

    ball_cand_score = []

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)
    # print(type(stats))

    if len(stats): 
        stats = np.delete(stats, 0, axis = 0)
        centroids = np.delete(centroids, 0, axis = 0)

    for i in range(len(stats)):
        x, y, w, h, area = stats[i]

        score = np.mean(pred_image[y:y+h, x:x+w])

        ball_cand_score.append(score)

    ball_pos = stats[np.argmax(ball_cand_score)]
    x_cen, y_cen = centroids[np.argmax(ball_cand_score)]

    x, y, w, h, area = ball_pos

    new_cen_x = int(x_cen * ratio_w)
    new_cen_y = int(y_cen * ratio_h)

    cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
    cv2.circle(image_ori, (new_cen_x, new_cen_y),  3, (0,0,255), -1)

    return image_ori

def find_ball_v3(pred_image, image_ori, depth_ori, ratio_w, ratio_h):

    ball_cand_score = []
    ball_cand_pos = []
    depth_list = []

    depth_check_array = np.nan_to_num(depth_ori, nan = 100)

    if np.amax(pred_image) <= 0: #no ball
        return image_ori, depth_list, ball_cand_pos, ball_cand_score

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)

    for i in range(len(stats)):
        x, y, w, h, area = stats[i]
        x_cen, y_cen = centroids[i]

        if area > 140000 or area < 5:
            continue

        x_0, x_1, y_0, y_1 = int(x * ratio_w), int((x + w) * ratio_w), int(y * ratio_h), int((y + h) * ratio_h)

        new_cen_x = int(x_cen * ratio_w)
        new_cen_y = int(y_cen * ratio_h)

        score = np.mean(pred_image[y:y+h, x:x+w])

        depth = np.min(depth_check_array[y_0 : y_1, x_0 : x_1]) + 0.03
        #depth = depth_check_array[new_cen_y,new_cen_x] - 0.5

        ball_cand_score.append(score)

        ball_cand_pos.append([new_cen_x, new_cen_y])

        depth_list.append(depth)

        cv2.rectangle(image_ori, (int(x * ratio_w), int(y * ratio_h)), (int((x + w) * ratio_w), int((y + h) * ratio_h)), (255,0,0), 3)
        cv2.circle(image_ori, (new_cen_x, new_cen_y),  3, (0,0,255), -1)

    return image_ori, depth_list, ball_cand_pos, ball_cand_score

def cal_ball_pos(ball_cand_pos, depth_list):

    focal_length = 319.9988245765257
    cx, cy = 320.5, 180.5

    ball_pos =[]

    for i in range(len(ball_cand_pos)):
        x_img, y_img = ball_cand_pos[i]

        x = depth_list[i]
        y = (x_img - cx)/focal_length * x
        z = (cy - y_img) * np.sqrt(x**2 + y**2)/np.sqrt(focal_length**2 + (x_img - cx)**2)

        ball_pos = [x,y,z]

    return ball_pos

def tran_input_img(img_list):

    trans_img = []

    #for i in reversed(range(len(img_list))):
    for i in range(len(img_list)):

        img = img_list[i]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = cv2.resize(img,(WIDTH, HEIGHT))
        img = np.asarray(img).transpose(2, 0, 1) / 255.0

        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])

    trans_img = np.asarray(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])


def tran_input_tensor(img_data, device):

    trans_img = []

    img_list = torch.from_numpy(np.array(img_data)).to(device, dtype=torch.float)

    for i in range(len(img_list)):

        img = img_list[i]

        img = img.flip(-1)

        img = img.permute(2, 0, 1) 

        trans_img.append(img[0])
        trans_img.append(img[1])
        trans_img.append(img[2])
    
    trans_img = torch.stack(trans_img)

    return trans_img.reshape(1,trans_img.shape[0],trans_img.shape[1],trans_img.shape[2])


def ball_segmentation(image_ori, image_pred, width, height):

    """ret, y_pred = cv2.threshold(image_pred,50,255, cv2.THRESH_BINARY)
    y_pred_rgb = cv2.cvtColor(y_pred, cv2.COLOR_GRAY2RGB)

    y_pred_rgb[...,0] = 0
    y_pred_rgb[...,1] = 0

    y_pred_rgb = cv2.resize(y_pred_rgb,(width, height))
    y_pred_rgb = cv2.resize(y_jet,(width, height))
    img = cv2.addWeighted(image_ori, 1, y_pred_rgb, 0.8, 0)
    """

    y_jet = cv2.applyColorMap(image_pred, cv2.COLORMAP_JET)
    y_jet = cv2.resize(y_jet,(width, height))

    img = cv2.addWeighted(image_ori, 1, y_jet, 0.3, 0)

    return img

def ball_vel_check(ball_trajectory):
    
    dT = 1/30
    
    ball_trajectory = np.array(ball_trajectory).reshape([-1,3])

    x_pos_list = ball_trajectory[-3:,0]
    y_pos_list = ball_trajectory[:,1]

    mean_x_vel = np.mean(np.diff(x_pos_list))/ dT
    mean_y_vel = np.mean(np.diff(y_pos_list))/ dT

    x_vel = ( ball_trajectory[-1][0] - ball_trajectory[-2][0]) / dT
    y_vel = ( ball_trajectory[-1][1] - ball_trajectory[-2][1]) / dT

    """if abs(abs(mean_x_vel) - abs(x_vel)) > 5: # 이전 x축 속도평균 보다 현재 속도가 5이상 더 빠를때
        ball_pos[0] = ball_trajectory[-1][0] + mean_x_vel * self.dT

    if abs(abs(mean_y_vel) - abs(y_vel)) > 1: # 이전 y축 속도평균 보다 현재 속도가 1이상 더 빠를때

        ball_pos[1] = ball_trajectory[-1][1] + mean_y_vel * self.dT

    if x_pos_list[-1] >= ball_pos[0]: #공이 뒤로 움직일때 이전 x 위치에서 평균 속도로 현재 위치 추정

        ball_pos[0] = ball_trajectory[-1][0] + mean_x_vel * self.dT"""

    return x_vel, y_vel