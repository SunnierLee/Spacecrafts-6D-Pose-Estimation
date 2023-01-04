import json
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

gt_rt_path = r'..\Wireframe Model Recovery\3D_Keypoints_Recovery\P2D\train.json'
gt_p2d_path = r'..\Wireframe Model Recovery\2D_Reproject\K2D_uv.json'
with open(gt_rt_path, 'r') as f:
    gt_rt = json.load(f)
with open(gt_p2d_path, 'r') as f:
    gt_p2d = json.load(f)


def get_error(image_name_, r_, t_):
    global gt_rt
    for gt in gt_rt:
        if gt['filename'] == image_name_:
            break
    r_gt = np.array(gt['q_vbs2tango'][-3:]) / gt['q_vbs2tango'][0]
    t_gt = np.array(gt['r_Vo2To_vbs_true'])
    error_position = np.linalg.norm(t_gt - t_) / np.linalg.norm(t_)
    # error_position = error_position if error_position >= 0.002173 * np.linalg.norm(t_gt) else 0
    error_orientation = 2 * np.arccos(np.abs(np.dot(r_gt, r_) / (np.linalg.norm(r_gt) * np.linalg.norm(r_))))
    # error_orientation = error_orientation if error_orientation >= 0.169 * 2 * np.pi / 360 else 0
    return error_position, error_orientation

difficult_str = 'img007499.jpg'
bad_bygt = []
bad_bydet = ['img009756.jpg', 'img012553.jpg', 'img012157.jpg']

regression_test = 'regression-based/points_by_regression_test_ori_dataaug_best_bydet_gooddet.txt'
detection_test = r'..\Satelite Detection\satellite_regression_test_best.txt'

with open(regression_test, 'r') as f:
    regression_lines = f.readlines()
with open(detection_test, 'r') as f:
    detection_lines = f.readlines()
points = []
boxes_gt = []
boxes_det = []
for line in detection_lines:
    line = line.strip().split(' ')
    boxes_gt.append(eval(line[1])[:4])
    if len(line) == 8:
        ymin, xmin, ymax, xmax = line[-4:]
    else:
        xmin = ymin = 0
        xmax = 1920
        ymax = 1200
    # ymin, xmin, ymax, xmax = line[-4:]
    # xmin, ymin, xmax, ymax = eval(line[-1])[:4]
    boxes_det.append([int(xmin), int(ymin), int(xmax), int(ymax)])
for line in regression_lines:
    line = line.strip().split(' ')
    points.append(line[-22:])
boxes = boxes_det
ori_path = r'D:\LargeData\PoseEstimation2019\speed\speed\images\real'
crop_path = r'D:\LargeData\PoseEstimation2019\speed\speed\images\crop_by_gt\train'
points_r = [1, 6, 6, 1, 5, 1, 1]
points_t = [2, 10, 10, 2, 8, 2, 2]
points_r_real = [6, 6, 6, 6, 6, 1, 1]
points_t_real= [10, 10, 10, 10, 10, 2, 2]
c = 0
for i in range(len(points)):
    point = points[i]
    point = np.array(point, dtype='float')
    box = boxes[i]
    # if box[-1] == 1:
    #     continue
    image_name = detection_lines[i].split(' ')[0].split('/')[-1]
    # ori_image = cv2.imread(os.path.join(ori_path, image_name))
    new_w, new_h = 224, 224
    xmin, ymin, xmax, ymax = box[:4]
    xmin_, ymin_, xmax_, ymax_ = box[:4]
    crop_w = xmax_ - xmin_
    crop_h = ymax_ - ymin_
    wh_ratio = 0.5
    if crop_w > crop_h:
        ymin_ -= (crop_w - crop_h) * wh_ratio
        ymax_ += (crop_w - crop_h) * (1 - wh_ratio)
        xmin_ -= (xmax_ - xmin_) * 0.2 / 2
        xmax_ += (xmax_ - xmin_) * 0.2 / 2
        ymin_ -= (ymax_ - ymin_) * 0.2 / 2
        ymax_ += (ymax_ - ymin_) * 0.2 / 2
    else:
        xmin_ -= (crop_h - crop_w) * wh_ratio
        xmax_ += (crop_h - crop_w) * (1 - wh_ratio)
        xmin_ -= (xmax_ - xmin_) * 0.2 / 2
        xmax_ += (xmax_ - xmin_) * 0.2 / 2
        ymin_ -= (ymax_ - ymin_) * 0.2 / 2
        ymax_ += (ymax_ - ymin_) * 0.2 / 2
    xmin_ = int(max(0, xmin_))
    ymin_ = int(max(0, ymin_))
    xmax_ = int(min(xmax_, 1920))
    ymax_ = int(min(ymax_, 1200))
    # crop_image = ori_image[ymin_:ymax_, xmin_:xmax_]
    crop_w = xmax_ - xmin_
    crop_h = ymax_ - ymin_
    dx_ = xmin - xmin_
    dy_ = ymin - ymin_
    scale = min(new_w / crop_w, new_h / crop_h)
    nw = int(crop_w * scale)
    nh = int(crop_h * scale)
    dx = (new_w - nw) // 2
    dy = (new_h - nh) // 2
    left, top = xmin_, ymin_

    for j in range(len(point) // 2):
        point[j*2] = (float(point[j*2]) - dx) * crop_w / nw + float(left)
        point[j*2+1] = (float(point[j*2+1]) - dy) * crop_h / nh + float(top)
        # point[j * 2] = (float(point[j * 2]) - dx) * crop_w / nw
        # point[j * 2 + 1] = (float(point[j * 2 + 1]) - dy) * crop_h / nh

    # if image_name in bad_bydet:
    #     ori_image = cv2.imread(os.path.join(ori_path, image_name))
    #     crop_image = ori_image[ymin_:ymax_, xmin_:xmax_]
    #     crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    #     fig = plt.figure()
    #     ax = fig.subplots()
    #     ax.imshow(crop_image)
    #     # cv2.imshow('x', crop_image)
    #     # cv2.waitKey(0)
    #     for j in range(len(point)//2):
    #         ax.scatter(int(point[j*2]), int(point[j*2+1]), color=(0, 1, 1))
    #     ax.axis('off')
    #     fig.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\pic/bad_reg_{}'.format(image_name.replace('jpg', 'pdf')))
    c += 1
    points[i] = point


'''获取关键点回归误差'''
# point_nums = 0
# error_sum = 0
# for i in range(len(regression_gt)):
#     reg_gt = regression_gt[i]
#     reg_pred = regression_pred[i]
#     for j in range(11):
#         point_class = 'K{}'.format(str(j+1))
#         if reg_gt[point_class] is None or reg_pred[point_class] is None:
#             continue
#         error_sum += np.sqrt((reg_gt[point_class][0]-reg_pred[point_class][0])**2+(reg_gt[point_class][1]-reg_pred[point_class][1])**2)
#         point_nums += 1
# print(error_sum/point_nums)


'''根据预测的关键点计算出位姿'''
pred_pose = []
threshold = 0.5
# original p3d
# point3s = np.array(
#                    [[0.306, -0.58, 0.254],
#                     [0.544, 0.490, 0.254],
#                     [-0.54, 0.487, 0.253],
#                     [0.37, -0.383, 0.321],
#                     [0.3635, 0.385, 0.317],
#                     [-0.3658, 0.3845, 0.32],
#                     [-0.3655, -0.384, 0.321],
#                     [0.368, -0.264, 0.0003],
#                     [0.3686, 0.3033, -0.00135],
#                     [-0.367, 0.301, -0.00182],
#                     [-0.368, -0.264, 0.00116]], dtype=np.double)
point3s = np.array(
    [[0.3211049, -0.57903574, 0.277346],
     [0.53806485, 0.49078455, 0.22039511],
     [-0.54734831, 0.4800094, 0.26518803],
     [0.40216169, -0.38423358, 0.31971899],
     [0.37414798, 0.38595157, 0.3063239],
     [-0.37047066, 0.37738984, 0.32095751],
     [-0.3589728, -0.39119226, 0.32030392],
     [0.37091823, -0.25966015, 0.00619306],
     [0.35495034, 0.30056408, -0.02179307],
     [-0.36899695, 0.30199781, -0.00175701],
     [-0.37242372, -0.26099563, -0.00444472]], dtype=np.double)


class Camera:
    def __init__(self):
        self.f_x = float(0.0176)  # m
        self.f_y = float(0.0176)  # m
        self.cx = float(960)
        self.cy = float(600)
        self.du = float(5.86e-6)  # m
        self.dv = float(5.86e-6)  # m
        self.fx = self.f_x/self.du
        self.fy = self.f_y/self.dv

    def K(self):
        k = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]])
        return k


camera = Camera()
dist = None

for j in range(len(points)):
    # if eval(boxes[j][-1])[-1] == 1:
    #     continue
    mask = []
    point = np.array(points[j], dtype=np.double).reshape((-1, 2))
    # for p in point:
    #     if p[0] < 0 or p[0] > 1920 or p[1] < 0 or p[1] > 1200:
    #         mask.append(False)
    #     else:
    #         mask.append(True)
    # mask = np.array(mask, dtype=np.bool)
    # point3s_ = point3s[mask]
    # point = point[mask]
    found, r, t = cv2.solvePnP(point3s, point, camera.K(), dist, flags=cv2.SOLVEPNP_EPNP)
    image_name = detection_lines[j].split(' ')[0].split('/')[-1]
    if image_name in bad_bydet:
        continue
    # '''重投影'''
    # image = cv2.imread(os.path.join(ori_path, image_name))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # fig = plt.figure()
    # ax = fig.subplots()
    # ax.imshow(image)
    # reproject_p2d, _ = cv2.projectPoints(point3s, r, t, camera.K(), None)
    # reproject_p2d = reproject_p2d.reshape(-1, 2)
    # color = (233 / 255, 85 / 255, 19 / 255)
    # line_width = 3
    # alpha = 0.8
    # '''绘制线框'''
    # ax.plot([reproject_p2d[0, 0], reproject_p2d[3, 0]], [reproject_p2d[0, 1], reproject_p2d[3, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[1, 0], reproject_p2d[4, 0]], [reproject_p2d[1, 1], reproject_p2d[4, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[2, 0], reproject_p2d[5, 0]], [reproject_p2d[2, 1], reproject_p2d[5, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[3, 0], reproject_p2d[4, 0]], [reproject_p2d[3, 1], reproject_p2d[4, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[3, 0], reproject_p2d[6, 0]], [reproject_p2d[3, 1], reproject_p2d[6, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[3, 0], reproject_p2d[7, 0]], [reproject_p2d[3, 1], reproject_p2d[7, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[4, 0], reproject_p2d[5, 0]], [reproject_p2d[4, 1], reproject_p2d[5, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[4, 0], reproject_p2d[8, 0]], [reproject_p2d[4, 1], reproject_p2d[8, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[5, 0], reproject_p2d[6, 0]], [reproject_p2d[5, 1], reproject_p2d[6, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[5, 0], reproject_p2d[9, 0]], [reproject_p2d[5, 1], reproject_p2d[9, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[6, 0], reproject_p2d[10, 0]], [reproject_p2d[6, 1], reproject_p2d[10, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[7, 0], reproject_p2d[8, 0]], [reproject_p2d[7, 1], reproject_p2d[8, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[7, 0], reproject_p2d[10, 0]], [reproject_p2d[7, 1], reproject_p2d[10, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[8, 0], reproject_p2d[9, 0]], [reproject_p2d[8, 1], reproject_p2d[9, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.plot([reproject_p2d[9, 0], reproject_p2d[10, 0]], [reproject_p2d[9, 1], reproject_p2d[10, 1]],
    #         color=color, linewidth=line_width, alpha=alpha)
    # ax.axis('off')
    # fig.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\pic/real_reg_project_{}'.format(
    #     image_name.replace('jpg', 'pdf')))
    pred_pose.append({'image_name': image_name, 'r': r, 't': t})
    print(r, t)


'''计算score'''
score_position = 0
score_orientation = 0
with open(gt_rt_path, 'r') as f:
    gt_rt = json.load(f)

error_positions_easy = []
error_positions_diff = []
errors = []
error_orientations_easy = []
error_orientations_diff = []
error_positions = []
error_orientations = []
for pred in pred_pose:
    image_name = pred['image_name']
    r_est = pred['r'].reshape(-1)
    t_est = pred['t'].reshape(-1)
    for gt in gt_rt:
        if gt['filename'] == image_name:
            break
    r_gt = np.array(gt['q_vbs2tango'][-3:]) / gt['q_vbs2tango'][0]
    t_gt = np.array(gt['r_Vo2To_vbs_true'])
    error_position, error_orientation = get_error(image_name, r_est, t_est)
    error_positions.append([error_position, np.linalg.norm(t_gt)])
    # error_positions.append(error_position)
    error_orientations.append([error_orientation, np.linalg.norm(t_gt)])
    # error_orientations.append(error_orientation)
    errors.append(error_orientation + error_position)
    score_position += error_position
    score_orientation += error_orientation
    if image_name > difficult_str:
        error_positions_diff.append(error_position)
        error_orientations_diff.append(error_orientation)
    else:
        error_positions_easy.append(error_position)
        error_orientations_easy.append(error_orientation)
error_positions = np.array(sorted(error_positions, key=lambda x: x[1]))
error_orientations = np.array(sorted(error_orientations, key=lambda x: x[1]))
# np.savetxt('position_error_byreg.csv', error_positions, delimiter=',')
# np.savetxt('orientation_error_byreg.csv', error_orientations, delimiter=',')
print(' %.4f & %.4f & %.4f & %.4f & %.4f & %.4f '%(np.median(error_positions[:, 0]), np.median(error_orientations[:, 0]), np.median(errors), np.mean(error_positions[:, 0]), np.mean(error_orientations[:, 0]), np.mean(errors)))
print('easy num:{}, error pose:{}, error orien:{}, score:{}'.format(len(error_orientations_easy), np.mean(error_positions_easy),
                                                                    np.mean(error_orientations_easy), np.mean(error_positions_easy)+np.mean(error_orientations_easy)))
print('difficult num:{}, error pose:{}, error orien:{}, score:{}'.format(len(error_orientations_diff), np.mean(error_positions_diff),
                                                                    np.mean(error_orientations_diff), np.mean(error_positions_diff)+np.mean(error_orientations_diff)))

    # r_gt = gt[]










