# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-02-05 21:16
# @Author : L_PenguinQAQ
# @File : extra_modules
# @Software: PyCharm
# @function: 自定义功能模块

import cv2
import numpy as np
import torch
import os


def show_img(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        cv2.imshow('result', img)
    else:
        c = img.shape[0]
        for i in range(c):
            for j in range(c):
                cv2.imshow(f'result_{i}_{j}', img[i][j].numpy())
    cv2.waitKey(0)


def seg_img(img, c=1, stride=32):
    """将图像分割返回输出
    Arguments:
        img: 图像输入，图像输入固定为等宽高的图像
        c: 图像分块个数
    Return:
        返回ndarray格式的数据，shape为(c, c)
    """
    if c == 1:
        return img
    else:
        result = np.array([[None for _ in range(c)] for i in range(c)])
        _, _, w, h = img.shape
        s_w = w // c
        s_h = h // c
        for i in range(c):
            for j in range(c):
                result[i][j] = img[:, :, i*s_w:(i+1)*s_w, j*s_h:(j+1)*s_h]

        return result


def judge_seg(prediction, conf_thres=0.25):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    xc = prediction[..., 4] > conf_thres

    return True in xc[0]


if __name__ == '__main__':
    path = f'./img.png'
    img = cv2.imread(path)
    img = torch.from_numpy(img)
    imgs = seg_img(img, c=2)
    show_img(imgs)
