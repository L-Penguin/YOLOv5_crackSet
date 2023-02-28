# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023/1/17 19:10
# @Author : L_PenguinQAQ
# @File : img_concat.py
# @Software: PyCharm
# @function: 图像拼接


import os
import cv2
import argparse
import numpy as np


def showPic(p, n):
    cv2.imshow(n, p)
    cv2.waitKey(0)

    cv2.imwrite(f'{n}/concat_img.jpg', p)


def pic_concat(img_arr):
    varr = [np.hstack(arr) for arr in img_arr]
    concat_img = np.vstack(varr)

    return concat_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--concat_path', default='concat_0000', type=str, help='input images path')
    opt = parser.parse_args()

    path = opt.concat_path
    img_arr = os.listdir(path)
    arr_t = [os.path.splitext(f)[0].split("-") for f in img_arr]
    x = max([int(i[1]) for i in arr_t])
    y = max([int(i[0]) for i in arr_t])

    arr = [[None for _ in range(int(x)+1)] for _ in range(int(y)+1)]

    for n in img_arr:
        full_path = os.path.join(path, n)
        img = cv2.imread(full_path)

        index_x = int(os.path.splitext(n)[0].split("-")[1])
        index_y = int(os.path.splitext(n)[0].split("-")[0])

        arr[index_y][index_x] = img

    concat_img = pic_concat(arr)

    showPic(concat_img, path)

