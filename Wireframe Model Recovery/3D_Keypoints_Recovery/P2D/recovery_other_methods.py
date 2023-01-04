#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import numpy as np
import random
from scipy.optimize import dual_annealing

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


def get_p2d_data(path):
    name = []
    p2df = []
    with open(path, 'r') as f:
        data = json.load(f)
    for i in data:
        name.append(i)
        p2df.append(data[i])
    return name, p2df


def get_p3d_data(path, p2d_name):
    p3df_q = []
    p3df_r = []
    with open(path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        if len(p3df_q) == len(p2d_name):
            break
        if data[i]['filename'] not in p2d_name:
            continue
        p3df_q.append(data[i]['q_vbs2tango'])
        p3df_r.append(data[i]['r_Vo2To_vbs_true'])
    return p3df_q, p3df_r


def q_to_Rt(q, t):
    Rt0 = []
    for i in range(len(q)):
        w = q[i][0]
        x = q[i][1]
        y = q[i][2]
        z = q[i][3]
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
            Rti[j][3] = t[i][j]
        Rt0.append(Rti)
    return Rt0


def object(v):
    global B
    global count
    global p2d
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12 = v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14]
    x, y, z = v[0], v[1], v[2]
    los1 = np.linalg.norm(s1*np.array([p2d[count[0]][0], p2d[count[0]][1], 1.0]).reshape([3, 1])-np.dot(B[count[0]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los2 = np.linalg.norm(s2*np.array([p2d[count[1]][0], p2d[count[1]][1], 1.0]).reshape([3, 1])-np.dot(B[count[1]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los3 = np.linalg.norm(s3*np.array([p2d[count[2]][0], p2d[count[2]][1], 1.0]).reshape([3, 1])-np.dot(B[count[2]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los4 = np.linalg.norm(s4*np.array([p2d[count[3]][0], p2d[count[3]][1], 1.0]).reshape([3, 1])-np.dot(B[count[3]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los5 = np.linalg.norm(s5*np.array([p2d[count[4]][0], p2d[count[4]][1], 1.0]).reshape([3, 1])-np.dot(B[count[4]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los6 = np.linalg.norm(s6*np.array([p2d[count[5]][0], p2d[count[5]][1], 1.0]).reshape([3, 1])-np.dot(B[count[5]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los7 = np.linalg.norm(s7*np.array([p2d[count[6]][0], p2d[count[6]][1], 1.0]).reshape([3, 1])-np.dot(B[count[6]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los8 = np.linalg.norm(s8*np.array([p2d[count[7]][0], p2d[count[7]][1], 1.0]).reshape([3, 1])-np.dot(B[count[7]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los9 = np.linalg.norm(s9*np.array([p2d[count[8]][0], p2d[count[8]][1], 1.0]).reshape([3, 1])-np.dot(B[count[8]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los10 = np.linalg.norm(s10*np.array([p2d[count[9]][0], p2d[count[9]][1], 1.0]).reshape([3, 1])-np.dot(B[count[9]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los11 = np.linalg.norm(s11*np.array([p2d[count[10]][0], p2d[count[10]][1], 1.0]).reshape([3, 1])-np.dot(B[count[10]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los12 = np.linalg.norm(s12*np.array([p2d[count[11]][0], p2d[count[11]][1], 1.0]).reshape([3, 1])-np.dot(B[count[11]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    fx = los1+los2+los3+los4+los5+los6+los7+los8+los9+los10+los11+los12
    return fx


def loss(count):
    global B
    global p2d
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12 = v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14]
    x, y, z = v[0], v[1], v[2]
    los1 = np.linalg.norm(s1*np.array([p2d[count[0]][0], p2d[count[0]][1], 1.0]).reshape([3, 1])-np.dot(B[count[0]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los2 = np.linalg.norm(s2*np.array([p2d[count[1]][0], p2d[count[1]][1], 1.0]).reshape([3, 1])-np.dot(B[count[1]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los3 = np.linalg.norm(s3*np.array([p2d[count[2]][0], p2d[count[2]][1], 1.0]).reshape([3, 1])-np.dot(B[count[2]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los4 = np.linalg.norm(s4*np.array([p2d[count[3]][0], p2d[count[3]][1], 1.0]).reshape([3, 1])-np.dot(B[count[3]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los5 = np.linalg.norm(s5*np.array([p2d[count[4]][0], p2d[count[4]][1], 1.0]).reshape([3, 1])-np.dot(B[count[4]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los6 = np.linalg.norm(s6*np.array([p2d[count[5]][0], p2d[count[5]][1], 1.0]).reshape([3, 1])-np.dot(B[count[5]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los7 = np.linalg.norm(s7*np.array([p2d[count[6]][0], p2d[count[6]][1], 1.0]).reshape([3, 1])-np.dot(B[count[6]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los8 = np.linalg.norm(s8*np.array([p2d[count[7]][0], p2d[count[7]][1], 1.0]).reshape([3, 1])-np.dot(B[count[7]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los9 = np.linalg.norm(s9*np.array([p2d[count[8]][0], p2d[count[8]][1], 1.0]).reshape([3, 1])-np.dot(B[count[8]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los10 = np.linalg.norm(s10*np.array([p2d[count[9]][0], p2d[count[9]][1], 1.0]).reshape([3, 1])-np.dot(B[count[9]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los11 = np.linalg.norm(s11*np.array([p2d[count[10]][0], p2d[count[10]][1], 1.0]).reshape([3, 1])-np.dot(B[count[10]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    los12 = np.linalg.norm(s12*np.array([p2d[count[11]][0], p2d[count[11]][1], 1.0]).reshape([3, 1])-np.dot(B[count[11]], np.array([x, y, z, 1.0]).reshape([4, 1])))
    fx = los1+los2+los3+los4+los5+los6+los7+los8+los9+los10+los11+los12
    return fx


path_2d = './K11.json'
path_3d = 'train.json'
camera = Camera()
B = []
p2d_name, p2d = get_p2d_data(path_2d)
p3d_q, p3d_r = get_p3d_data(path_3d, p2d_name)
Rt = q_to_Rt(p3d_q, p3d_r)
for i in range(24):
    B.append(np.dot(camera.K(), Rt[i]))
# 优化参数设置
r_min, r_max = -1, 1
s_min, s_max = 0, 30

res = []
list = []
for i in range(24):
    list.append(i)
bounds = [[r_min, r_max], [r_min, r_max], [r_min, r_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max], [s_min, s_max]]
while True:
    count = random.sample(list, 12)
    count = np.arange(24)
    A = []
    b = []
    for idx in count:
        u = p2d[idx][0]
        v = p2d[idx][1]
        P = B[idx]
        A.append([u * P[2, 0] - P[0, 0], u * P[2, 1] - P[0, 1], u * P[2, 2] - P[0, 2]])
        A.append([v * P[2, 0] - P[1, 0], u * P[2, 1] - P[1, 1], u * P[2, 2] - P[1, 2]])
        b.append([P[0, 3] - u * P[2, 3]])
        b.append([P[1, 3] - v * P[2, 3]])
    A = np.array(A)
    b = np.array(b)
    s = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), b)
    print(count)
    print(s.reshape(-1))
    break