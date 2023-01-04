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

        '''获取预测的 crop image'''
        test_txt = 'satellite_detection_test.txt'
        image_save_path = '/media/group1/lhj/data_satellite/speed/crop/test/'
        line_save_path = 'satellite_detection_test_2021.txt'
        with open(test_txt, 'r') as f:
            test_lines = f.readlines()
        with open(line_save_path, 'w') as f:
            for test_line in test_lines:
                image_name = test_line.split(' ')[0].split('/')[-1]
                image = Image.open(test_line.split(' ')[0])
                r_image, lines = yolo.detect_image(image)
                # top, left, bottom, right = lines.strip().split(' ')[-4:]
                # top = int(top)
                # left = int(left)
                # bottom = int(bottom)
                # right = int(right)
                # image_save = cv2.imread(test_line.split(' ')[0])
                # top = max(0, top - 20)
                # left = max(0, left - 20)
                # bottom = min(image_save.shape[0], bottom+20)
                # right = min(image_save.shape[1], right+20)
                # image_save = image_save[top:bottom, left:right]
                # cv2.imwrite(os.path.join(image_save_path, image_name), image_save)
                line_save = test_line.strip() + lines + '\n'
                f.write(line_save)
