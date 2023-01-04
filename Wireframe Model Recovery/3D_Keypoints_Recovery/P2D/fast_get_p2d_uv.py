#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import json
import os

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


def q_to_Rt0(q, t):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    Rti = np.arange(12, dtype=np.float64).reshape([3, 4])
    Rti[0, 0] = 1-2*y*y-2*z*z
    Rti[0, 1] = 2*x*y-2*z*w
    Rti[0, 2] = 2*x*z+2*y*w
    Rti[1, 0] = 2*x*y+2*z*w
    Rti[1, 1] = 1-2*x*x-2*z*z
    Rti[1, 2] = 2*y*z-2*x*w
    Rti[2, 0] = 2*x*z-2*y*w
    Rti[2, 1] = 2*y*z+2*x*w
    Rti[2, 2] = 1-2*x*x-2*y*y
    for j in range(3):
        Rti[j][3] = t[j]
    return Rti


path_p3d = r'D:\Large Data\Pose Estimation 2019\speed\speed\train.json'
path_recov_p3d = 'k_location.txt'
K_3D = {}
camera = Camera()
can_point = []
name = []
p3d_q = []
p3d_r = []
with open(path_recov_p3d, 'r') as f:
    data = f.readlines()
    for i in data:
        Ki = i.split(' ')[0]
        K_xyz = eval(i.split(' ')[1])
        K_3D[Ki] = K_xyz
with open(path_p3d, 'r') as f:
    data = json.load(f)
    for i in data:
        name.append(i['filename'])
        p3d_q.append(i['q_vbs2tango'])
        p3d_r.append(i['r_Vo2To_vbs_true'])
K_info = {}
for img_name in name:
    K_2D = {}
    img = cv2.imread(os.path.join(r'D:\Large Data\Pose Estimation 2019\speed\speed\images\train', img_name))
    img_index = name.index(img_name)
    rt = q_to_Rt0(p3d_q[img_index], p3d_r[img_index])
    rt = rt.flatten().copy()
    for i in K_3D.keys():
        x, y, z = K_3D[i]
        u = (x*camera.fx*rt[0]+x*camera.cx*rt[8]+
            y*camera.fx*rt[1]+y*camera.cx*rt[9]+
            z*camera.fx*rt[2]+z*camera.cx*rt[10]+
            camera.fx*rt[3]+camera.cx*rt[11])/(x*rt[8]+y*rt[9]+z*rt[10]+rt[11])
        v = (x*camera.fy*rt[4]+x*camera.cy*rt[8]+
            y*camera.fy*rt[5]+y*camera.cy*rt[9]+
            z*camera.fy*rt[6]+z*camera.cy*rt[10]+
            camera.fy*rt[7]+camera.cy*rt[11])/(x*rt[8]+y*rt[9]+z*rt[10]+rt[11])
        u, v = int(u), int(v)
        K_2D[i] = [u, v]
        xyz = "%f,%f,%f" % (x, y, z)
        cv2.circle(img, (u, v), 1, (0, 255, 255), thickness=3)
        cv2.putText(img, xyz, (u, v), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), thickness=1)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    K_info[img_name] = K_2D
# with open(r'C:\Users\11598\Desktop\pycharm\lab\asc2019\2D_Reproject\K2D_uv.json', 'w') as f:
#     json.dump(K_info, f)


