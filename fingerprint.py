import json
from itertools import product
from pprint import pprint

import numpy as np
from cv2 import cv2


def iter_block(grayscale_image, nh, nw):
    """
    迭代返回图像的某一分块，先按行方向，后按列方向，
    即如果按4×4分块图像，则第一次返回的是左上角的分块，第四次换回的是右上角的分块，
    最后一次返回的是右下角的分块。

    参数：
        grayscale_image: 需要进行分块的灰度图像，为2维ndarray对象。
        nh: 在图像列方向上需要分的块数。
        nw: 在图像行方向上需要分的块数。

    返回：
        图像的某一分块。
    """

    h, w = grayscale_image.shape
    sh, sw =  h // nh, w // nw
    for i in range(nh):
        start_h = i * sh
        end_h = (i + 1) * sh if i != nh - 1 else None
        for j in range(nw):
            start_w = j * sw
            end_w = (j + 1) * sw if j != nw - 1 else None
            yield grayscale_image[start_h:end_h, start_w:end_w]


def difference_value_hash(grayscale_image):
    """
    计算灰度图像的差值哈希值，计算方式为：
    首先将图像利用三次内卷插值法重采样为10×10的图像，
    然后取10×10图像最中间的8×8大小的一块，记为center，取位于10×10最左上角的8×8大小的一块，记为up_left
    用center减去up_left得到大小为8×8的差值矩阵，将差值矩阵中大于0的部分替换为1，小于等于0的部分替换为0，并展开拼接成一个长度为64位的0-1字符串，
    对up，up_right，right，down_right，down，down_left做类似的操作，
    最后得到8个64位字符串，将其拼接成长度为512位的字符串，表示图像的结构信息。

    参数：
        grayscale：需要计算差值哈希值的灰度图像。

    返回：
        图像的差值哈希值，为长度是512的0-1字符串。
    """

    # 重采样
    grayscale_image = cv2.resize(grayscale_image, (10, 10), interpolation=cv2.INTER_CUBIC)
    grayscale_image = grayscale_image.astype(np.int32)
    start_i, start_j, base_h, base_w = 1, 1, 8, 8

    # 获取位于最中心的8×8切片
    center = grayscale_image[start_i: start_i + base_h, start_j: start_j + base_w]
    hash_str = ''
    # 通过笛卡尔內积得到九个方向的矢量
    for offset_i, offset_j in product((-1, 0, 1), (-1, 0, 1)):
        # 如果是零矢量则跳过
        if offset_i == 0 and offset_j == 0:
            continue

        # 获取指定向量偏移下的切片，如(-1, -1)偏移可获得up_left切片
        offset = grayscale_image[start_i + offset_i: start_i + base_h + offset_i, start_j + offset_j: start_j + base_w + offset_j]

        # 计算差值矩阵
        difference_value_image = center - offset

        # 拼接哈希字符串
        hash_str += ''.join(map(lambda x: '1' if x > 0 else '0', difference_value_image.ravel()))
    return hash_str


def blocked_fingerprint(bgr_image, nh=4, nw=4, fingerprint_function_list=[np.mean, np.var, difference_value_hash]):
    """
    计算彩色图像所有区域指纹信息。

    参数：
        bgr_image: 彩色图像，像素的颜色顺序为B、G、R，为OpenCV默认的图像读取结果
        nh: 在图像列方向上需要分的块数。
        nw: 在图像行方向上需要分的块数。
        fingerprint_function_list: 计算指纹各个部分的函数，包括均值函数，方差函数，差值哈希函数。
    
    返回：
        图像所有区域指纹信息的列表。
    """

    # 将彩色图形转为灰度图像
    # 具体公式为：GRAY = R * 0.299 + G * 0.587 + B * 0.114
    grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    res = list()
    for block in iter_block(grayscale_image, nh, nw):
        # 计算单个区域的指纹信息
        block_fingerprint = list(func(block) for func in fingerprint_function_list)
        res.append(block_fingerprint)
    return res


def test():
    lena_std = cv2.imread('lena_std.tif')
    res = blocked_fingerprint(lena_std)
    pprint(res)


def main():
    test()


if __name__ == '__main__':
    main()
