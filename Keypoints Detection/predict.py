import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
import os

if __name__ == "__main__":
    yolo = YOLO()
    mode = "predict"

    if mode == "predict":

        '''获取预测的 points'''
        test_txt = 'satellite_detreg_test_best_withbydet.txt'
        image_save_path = 'satellite_regression_test_2021'
        line_save_path = 'satellite_regression_test_s224_pre_0threshold_299_best_bydet.txt'
        with open(test_txt, 'r') as f:
            test_lines = f.readlines()
        with open(line_save_path, 'w') as f:
            for test_line in test_lines:
                image_name = test_line.split(' ')[0].split('/')[-1]
                image = Image.open(test_line.split(' ')[0])
                #box = eval(test_line.split(' ')[-1])
                box = test_line.strip().split(' ')[-4:]
                #xmin, ymin, xmax, ymax = box[:4]
                ymin, xmin, ymax, xmax = box[:4]
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                xmin_, ymin_, xmax_, ymax_ = xmin, ymin, xmax, ymax
                wh_ratio = 0.5
                crop_w = xmax_ - xmin_
                crop_h = ymax_ - ymin_
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
                image = image.crop((int(xmin_), int(ymin_), int(xmax_), int(ymax_)))
            
                r_image, lines = yolo.detect_image(image)
                #r_image.save(os.path.join(image_save_path, image_name))
                line_save = test_line.strip() + lines + ' ' + str(xmin_) + ',' + str(ymin_) + ',' + str(xmax_) + ',' + str(ymax_) + ',' + '0' + '\n'
                f.write(line_save)
