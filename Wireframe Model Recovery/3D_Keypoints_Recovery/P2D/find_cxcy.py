#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import json


class Camera:
    def __init__(self):
        self.f_x = float(0.0176)  # m
        self.f_y = float(0.0176)  # m
        self.cx = float(930)
        self.cy = float(500)
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
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
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


good_camera = []
tr_baoli = q_to_Rt0([0.239245, 0.941466, 0.210435, 0.110098], [0.034544, 0.002258, 4.02238])
rt_check = q_to_Rt0([-0.179129, 0.689599, -0.700355, -0.043231], [0.065936, -0.037431, 4.680168])
tr_baoli = tr_baoli.flatten().copy()
rt_check = rt_check.flatten().copy()
camera = Camera()
for m in range(40):
    for n in range(20):
        camera.cx = 800+m*10
        camera.cy = 500+n*10
        can_point = []
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    x, y, z = -1.0 + 0.02 * i, -1.0 + 0.02 * j, -1.0 + 0.02 * k
                    u = (x * camera.fx * tr_baoli[0] + x * camera.cx * tr_baoli[8] +
                         y * camera.fx * tr_baoli[1] + y * camera.cx * tr_baoli[9] +
                         z * camera.fx * tr_baoli[2] + z * camera.cx * tr_baoli[10] +
                         camera.fx * tr_baoli[3] + camera.cx * tr_baoli[11]) / (x * tr_baoli[8] + y * tr_baoli[9] + z * tr_baoli[10] + tr_baoli[11])
                    v = (x * camera.fy * tr_baoli[4] + x * camera.cy * tr_baoli[8] +
                         y * camera.fy * tr_baoli[5] + y * camera.cy * tr_baoli[9] +
                         z * camera.fy * tr_baoli[6] + z * camera.cy * tr_baoli[10] +
                         camera.fy * tr_baoli[7] + camera.cy * tr_baoli[11]) / (x * tr_baoli[8] + y * tr_baoli[9] + z * tr_baoli[10] + tr_baoli[11])
                    u, v = int(u), int(v)
                    if (u-682)**2 + (v-611)**2 > 500:
                        continue
                    can_point.append([x, y, z])
        for i in can_point:
            x, y, z = i[0], i[1], i[2]
            u = (x * camera.fx * rt_check[0] + x * camera.cx * rt_check[8] +
                 y * camera.fx * rt_check[1] + y * camera.cx * rt_check[9] +
                 z * camera.fx * rt_check[2] + z * camera.cx * rt_check[10] +
                 camera.fx * rt_check[3] + camera.cx * rt_check[11]) / (x * rt_check[8] + y * rt_check[9] + z * rt_check[10] + rt_check[11])
            v = (x * camera.fy * rt_check[4] + x * camera.cy * rt_check[8] +
                 y * camera.fy * rt_check[5] + y * camera.cy * rt_check[9] +
                 z * camera.fy * rt_check[6] + z * camera.cy * rt_check[10] +
                 camera.fy * rt_check[7] + camera.cy * rt_check[11]) / (x * rt_check[8] + y * rt_check[9] + z * rt_check[10] + rt_check[11])
            u, v = int(u), int(v)
            if (u-1288)**2+(v-860)**2 <= 500:
                good_camera.append([camera.cx, camera.cy])
print(good_camera)