# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023/1/11 22:24
# @Author : L_PenguinQAQ
# @File : img_seg.py
# @Software: PyCharm
# @function: 将图像分割成若干227*227像素的子图


import cv2
import os
import argparse
import numpy as np


def showPic(p, n, d):
    cv2.imshow(n, p)
    cv2.waitKey(0)

    cv2.imwrite(f'./{d}/{n}', p)


def picture_seg(p, path, s=227):
    """将图片分割成w*w的若干子图

        Arguments:
            p: 输入图片
            s: 按照多少像素点分割子图

        Return:
            返回存放子图的数列

        Usage:
            from img_seg import picture_seg as ps
    """
    imgArr = []

    h, w, _ = p.shape
    if h < s or w < s:
        t_1 = s // h
        t_2 = s // w

        t = max(t_1, t_2) + 1
        img = cv2.resize(p, dsize=None, fx=t, fy=t, interpolation=cv2.INTER_LINEAR)
    else:
        img = p.copy()

    # 添加的宽和高的边框像素值
    b_h = s - (img.shape[0] % s)
    b_w = s - (img.shape[1] % s)

    if b_h == s:
        b_h = 0

    if b_w == s:
        b_w = 0

    img = cv2.copyMakeBorder(img, 0, b_h, 0, b_w, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h_img, w_img, _ = img.shape

    i_h, i_w = h_img // s, w_img // s

    dirName = "seg_" + os.path.splitext(path)[0] + f"_{i_h}-{i_w}"
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    for i in range(i_h):
        arr = []
        for j in range(i_w):
            img_temp = img[i*s:(i+1)*s, j*s:(j+1)*s]
            cv2.imwrite(f'{dirName}/{i}-{j}.jpg', img_temp)
            arr.append(img_temp)

        imgArr.append(arr)

    return imgArr, dirName


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', default='./0000jpg', type=str, help='input images path')
    parser.add_argument('--stride', default=40, type=int, help='stride path')
    opt = parser.parse_args()

    path = opt.imgs_path
    s = opt.stride
    pic = cv2.imread(path)
    img_arr, dirName = picture_seg(pic, path, s)

    varr = [np.hstack([cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 0, 0]) for img in arr]) for arr in img_arr]

    seg_img = np.vstack(varr)
    showPic(seg_img, path, dirName)


