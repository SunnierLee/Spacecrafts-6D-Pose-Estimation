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

new_path_1 = r'D:\LargeData\PoseEstimation2019\speed\speed\images\real'
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
    regression_pred.append(regression_template)


'''根据预测的关键点计算出位姿'''
score_t_100 = []
refuse_num = []
for threshold in range(1):
    threshold /= 100
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
        error_list = []
        for topk in [11, 7]:
            reg_pred = regression_pred[j]
            points_3d = []
            points_2d = []
            score = []
            image_name = detection_lines[j].split(' ')[0].split('/')[-1]
            for i in range(11):
                pred = list(reg_pred.values())[i+1]
                if pred is None or pred[2] < threshold:
                    continue
                if len(score) >= topk:
                    if pred[2] > min(score):
                        idx = np.argmin(score)
                        score.pop(idx)
                        points_3d.pop(idx)
                        points_2d.pop(idx)
                    else:
                        continue
                score.append(pred[2])
                points_3d.append(point3s[i])
                points_2d.append([pred[0], pred[1]])
            points_3d_array = np.array(points_3d, dtype=np.double)
            points_2d_array = np.array(points_2d, dtype=np.double)
            if points_2d_array.shape[0] > 4:
                found, r, t = cv2.solvePnP(points_3d_array, points_2d_array, camera.K(), dist, flags=cv2.SOLVEPNP_EPNP)
            else:
                refuse_pic.append(reg_pred['image_name'])
                # print(reg_pred['image_name'])
                continue
            a, b = get_error(image_name, r.reshape(-1), t.reshape(-1))
            error_list.append([a, b])
        if len(error_list) == 2 and sum(error_list[0]) - sum(error_list[1]) > 0.05:
            print(image_name, (error_list[0][0] - error_list[1][0])/error_list[0][0],
                  (error_list[0][1] - error_list[1][1])/error_list[0][1],
                  (sum(error_list[0]) - sum(error_list[1]))/sum(error_list[0]))
            # image = cv2.imread(os.path.join(new_path_1, image_name))
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # fig = plt.figure()
            # ax = fig.subplots()
            # ax.imshow(image)
            # for i in range(11):
            #     pred = list(reg_pred.values())[i + 1]
            #     if [pred[0], pred[1]] in points_2d:
            #         ax.scatter(int(pred[0]), int(pred[1]), color=(0, 1, 1))
            #     else:
            #         # gt_p2d_item = list(gt_p2d[image_name].values())
            #         # ax.scatter(int(gt_p2d_item[i][0]), int(gt_p2d_item[i][1]), color=(1, 1, 0))
            #         ax.scatter(int(pred[0]), int(pred[1]), color=(1, 0, 0))
            # ax.axis('off')
            # fig.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\pic\point_uncertainty/{}'.format(image_name.replace('jpg', 'pdf')))
        # '''重投影'''
        # image = cv2.imread(os.path.join(new_path_1, image_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # fig = plt.figure()
        # ax = fig.subplots()
        # ax.imshow(image)
        # reproject_p2d, _ = cv2.projectPoints(point3s, r, t, camera.K(), None)
        # reproject_p2d = reproject_p2d.reshape(-1, 2)
        # color = (233/255, 85/255, 19/255)
        # line_width = 3
        # alpha=0.8
        # '''绘制线框'''
        # ax.plot([reproject_p2d[0, 0], reproject_p2d[3, 0]], [reproject_p2d[0, 1], reproject_p2d[3, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[1, 0], reproject_p2d[4, 0]], [reproject_p2d[1, 1], reproject_p2d[4, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[2, 0], reproject_p2d[5, 0]], [reproject_p2d[2, 1], reproject_p2d[5, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[3, 0], reproject_p2d[4, 0]], [reproject_p2d[3, 1], reproject_p2d[4, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[3, 0], reproject_p2d[6, 0]], [reproject_p2d[3, 1], reproject_p2d[6, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[3, 0], reproject_p2d[7, 0]], [reproject_p2d[3, 1], reproject_p2d[7, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[4, 0], reproject_p2d[5, 0]], [reproject_p2d[4, 1], reproject_p2d[5, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[4, 0], reproject_p2d[8, 0]], [reproject_p2d[4, 1], reproject_p2d[8, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[5, 0], reproject_p2d[6, 0]], [reproject_p2d[5, 1], reproject_p2d[6, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[5, 0], reproject_p2d[9, 0]], [reproject_p2d[5, 1], reproject_p2d[9, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[6, 0], reproject_p2d[10, 0]], [reproject_p2d[6, 1], reproject_p2d[10, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[7, 0], reproject_p2d[8, 0]], [reproject_p2d[7, 1], reproject_p2d[8, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[7, 0], reproject_p2d[10, 0]], [reproject_p2d[7, 1], reproject_p2d[10, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[8, 0], reproject_p2d[9, 0]], [reproject_p2d[8, 1], reproject_p2d[9, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.plot([reproject_p2d[9, 0], reproject_p2d[10, 0]], [reproject_p2d[9, 1], reproject_p2d[10, 1]],
        #          color=color, linewidth=line_width, alpha=alpha)
        # ax.axis('off')
        # fig.savefig(r'C:\Users\Administrator\Desktop\python\local_python\Pose Estimation\pic/real_det_project_{}'.format(image_name.replace('jpg', 'pdf')))
        pred_pose.append({'image_name': reg_pred['image_name'], 'r': r, 't': t})
        # print(r, t)


    '''计算score'''
    score_position = 0
    score_orientation = 0
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
        error_position, error_orientation = get_error(image_name, r_est, t_est)
        # error_positions.append([error_position, np.linalg.norm(t_gt)])
        error_positions.append(error_position)
        # error_orientations.append([error_orientation, np.linalg.norm(t_gt)]).
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










