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

    if np.amax(pred_image) <= 0: #no ball
        return image_ori, depth_list, ball_cand_pos, ball_cand_score

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_image, connectivity = 8)

    for i in range(len(stats)):
        x, y, w, h, area = stats[i]
        x_cen, y_cen = centroids[i]

        if area > 150000:
            continue

        x_0, x_1, y_0, y_1 = int(x * ratio_w), int((x + w) * ratio_w), int(y * ratio_h), int((y + h) * ratio_h)

        new_cen_x = int(x_cen * ratio_w)
        new_cen_y = int(y_cen * ratio_h)

        score = np.mean(pred_image[y:y+h, x:x+w])

        depth = depth_ori[new_cen_y, new_cen_x]

        ball_cand_score.append(score)

        ball_cand_pos.append([x_0, x_1, y_0, y_1])

        depth_list.append(depth)


    return image_ori, depth_list, ball_cand_pos, ball_cand_score


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

        img = img.permute(2, 0, 1) / 255.0

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