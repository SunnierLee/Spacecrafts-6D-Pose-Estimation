import json
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

gt_rt_path = r'D:\LargeData\PoseEstimation2019\speed\speed\train.json'
gt_p2d_path = r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\zhanghaopeng\asc2019\2D_Reproject\K2D_uv.json'
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


def get_points2d_error(image_name_, p_, mask_):
    global gt_p2d
    for gt_image_name in gt_p2d:
        if gt_image_name == image_name_:
            break
    gt_p2d_one = np.array(list(gt_p2d[gt_image_name].values()))
    gt_p2d_one = gt_p2d_one[mask_]
    return np.mean(np.sqrt((gt_p2d_one - p_) ** 2))


difficult_str = 'img007499.jpg'

regression_test = 'satellite_regression_test_s224_0threshold_best_bydet_0179.txt'
detection_test = r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\zhanghaopeng\Satelite Detection\satellite_regression_test_best.txt'
bad_bygt = []
bad_bydet = ['img009756.jpg', 'img012553.jpg', 'img012157.jpg']
regression_gt = []
regression_pred = []
regression_template = {'K1': None, 'K2': None, 'K3': None, 'K4': None, 'K5': None,
                       'K6': None, 'K7': None, 'K8': None, 'K9': None, 'K10': None, 'K11': None}
detection_pred = []

with open(regression_test, 'r') as f:
    regression_lines = f.readlines()
with open(detection_test, 'r') as f:
    detection_lines = f.readlines()

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
boxes = boxes_det

new_path_1 = r'D:\LargeData\PoseEstimation2019\speed\speed\images\train'
count = 0
points_r = [1, 6, 6, 1, 5, 1, 1]
points_t = [2, 10, 10, 2, 8, 2, 2]
points_r_real = [6, 6, 6, 6, 6, 1, 1]
points_t_real = [10, 10, 10, 10, 10, 2, 2]
c = 0
for line in regression_lines:
    box = boxes[regression_lines.index(line)]
    line = line.strip().split(' ')
    image_name = line[0].split('/')[-1]

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
    right, bottom = xmax_, ymax_

    regression_template = {'image_name': None, 'K1': None, 'K2': None, 'K3': None, 'K4': None, 'K5': None,
                           'K6': None, 'K7': None, 'K8': None, 'K9': None, 'K10': None, 'K11': None}
    regression_template['image_name'] = image_name
    for i in range(len(line)):
        if 'b\'' in line[i]:
            point_class = line[i].split('\'')[1]
            if point_class == 'satellite':
                continue
            score = float(line[i+1].replace('\'', ''))
            y = (float(line[i+2]) + float(line[i+4]))/2 + top
            x = (float(line[i+3]) + float(line[i+5]))/2 + left
            # y = (float(line[i + 2]) + float(line[i + 4])) / 2
            # x = (float(line[i+3]) + float(line[i+5]))/2
            if regression_template[point_class] is None or regression_template[point_class][2] < score:
                regression_template[point_class] = [x, y, score]
    # if image_name in bad_bydet or True:
    #     image = cv2.imread(os.path.join(new_path_1, image_name))
    #     # image = image[top:bottom, left:right]
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     fig = plt.figure()
    #     ax = fig.subplots()
    #     ax.imshow(image_rgb)
    #     ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top, color="red", fill=False, linewidth=1))
    #     for point_class in regression_template:
    #         if point_class == 'image_name' or regression_template[point_class] is None or regression_template[point_class][2] < 0.0:
    #             continue
    #         if regression_template[point_class][2] > 0.5:
    #             color_p = (0, 1, 1)
    #         else:
    #             color_p = (1, 0, 0)
    #         x, y = regression_template[point_class][:2]
    #         x, y = int(x), int(y)
    #         ax.scatter(x, y, color=color_p, )
    #     plt.axis('off')
    #     plt.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\pic\point_uncertainty/{}'.format(image_name))
    #     # plt.show()
    #     c += 1
    regression_pred.append(regression_template)
    count += 1

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
for topk in range(7):
    score_t_100 = []
    refuse_num = []
    points_det_error = []
    threshold = 0.5
    pred_pose = []
    refuse_pic = []
    # threshold = 0.5
    point3s = np.array(
                       [[0.306, -0.58, 0.254],
                        [0.544, 0.490, 0.254],
                        [-0.54, 0.487, 0.253],
                        [0.37, -0.383, 0.321],
                        [0.3635, 0.385, 0.317],
                        [-0.3658, 0.3845, 0.32],
                        [-0.3655, -0.384, 0.321],
                        [0.368, -0.264, 0.0003],
                        [0.3686, 0.3033, -0.00135],
                        [-0.367, 0.301, -0.00182],
                        [-0.368, -0.264, 0.00116]], dtype=np.double)


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

    for j in range(len(regression_pred)):
        reg_pred = regression_pred[j]
        points_3d = []
        points_2d = []
        score = []
        image_name = detection_lines[j].split(' ')[0].split('/')[-1]
        idxs = []
        for i in range(11):
            pred = list(reg_pred.values())[i+1]
            if pred is None or pred[2] < threshold:
                continue
            if len(score) >= 11-topk:
                if pred[2] > min(score):
                    idx = np.argmin(score)
                    score.pop(idx)
                    points_3d.pop(idx)
                    points_2d.pop(idx)
                    idxs.pop(idx)
                else:
                    continue
            score.append(pred[2])
            idxs.append(i)
            points_3d.append(point3s[i])
            points_2d.append([pred[0], pred[1]])
        points_3d = np.array(points_3d, dtype=np.double)
        points_2d = np.array(points_2d, dtype=np.double)
        mask = [False for i in  range(11)]
        for i in idxs:
            mask[i] = True
        points_det_error.append(get_points2d_error(image_name, points_2d, mask))
        if points_2d.shape[0] > 4:
            found, r, t = cv2.solvePnP(points_3d, points_2d, camera.K(), dist, flags=cv2.SOLVEPNP_EPNP)
        else:
            refuse_pic.append(reg_pred['image_name'])
            # print(reg_pred['image_name'])
            continue
        pred_pose.append({'image_name': reg_pred['image_name'], 'r': r, 't': t})
        # print(r, t)
    print(np.mean(points_det_error))


    '''计算score'''
    score_position = 0
    score_orientation = 0
    gt_rt_path = r'D:\LargeData\PoseEstimation2019\speed\speed\train.json'
    with open(gt_rt_path, 'r') as f:
        gt_rt = json.load(f)

    error_positions = []
    error_orientations = []
    errors = []
    error_positions_easy = []
    error_positions_diff = []
    error_orientations_easy = []
    error_orientations_diff = []
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
        # error_positions.append([error_position, np.linalg.norm(t_gt)])
        error_positions.append(error_position)
        # error_orientations.append([error_orientation, np.linalg.norm(t_gt)])
        error_orientations.append(error_orientation)
        errors.append(error_orientation+error_position)
        score_position += error_position
        score_orientation += error_orientation
        if image_name > difficult_str:
            error_positions_diff.append(error_position)
            error_orientations_diff.append(error_orientation)
        else:
            error_positions_easy.append(error_position)
            error_orientations_easy.append(error_orientation)
    # error_positions = np.array(sorted(error_positions, key=lambda x: x[1]))
    # error_orientations = np.array(sorted(error_orientations, key=lambda x: x[1]))
    # np.savetxt('position_error_bydet.csv', error_positions, delimiter=',')
    # np.savetxt('orientation_error_bydet.csv', error_orientations, delimiter=',')
    if len(pred_pose) == 0:
        break
    # print(np.mean(error_positions), np.mean(error_orientations), np.mean(errors))
    # print(np.median(error_positions), np.median(error_orientations), np.median(errors))
    print(' %.4f & %.4f & %.4f & %.4f & %.4f & %.4f '%(np.median(error_positions), np.median(error_orientations), np.median(errors), np.mean(error_positions), np.mean(error_orientations), np.mean(errors)))
    print('easy num:{}, error pose:{}, error orien:{}, score:{}'.format(len(error_orientations_easy), np.mean(error_positions_easy),
                                                                        np.mean(error_orientations_easy), np.mean(error_positions_easy)+np.mean(error_orientations_easy)))
    print('difficult num:{}, error pose:{}, error orien:{}, score:{}'.format(len(error_orientations_diff), np.mean(error_positions_diff),
                                                                        np.mean(error_orientations_diff), np.mean(error_positions_diff)+np.mean(error_orientations_diff)))
    score_t_100.append(score)
    refuse_num.append(len(refuse_pic))
    # break
print(score_t_100)
print(refuse_num)

    # r_gt = gt[]










