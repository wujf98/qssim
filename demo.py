from cv2 import cv2

from qssim import mqssim
from fingerprint import blocked_fingerprint


def calc_qssim_from_file(fname_1, fname_2):
    # 读入图像
    im_1 = cv2.imread(fname_1)
    im_2 = cv2.imread(fname_2)
    
    # 计算图像所有区域的指纹信息
    res_1 = blocked_fingerprint(im_1)
    res_2 = blocked_fingerprint(im_2)

    # 计算匹配程度
    v = mqssim(res_1, res_2)
    print(fname_1, '<->', fname_2, ':', '{:.2f}%'.format(v * 100))
    return v


def test():
    # 完全相似
    calc_qssim_from_file('lena_std.jpg', 'lena_std.tif')

    # 缩放
    calc_qssim_from_file('lena_std.jpg', 'lena_std_256.jpg')

    # 水印
    calc_qssim_from_file('lena_std.jpg', 'lena_std_watermark.jpg')

    # 裁剪
    calc_qssim_from_file('lena_std.jpg', 'lena_hires.jpg')

    # 完全不相似
    calc_qssim_from_file('lena_std.jpg', 'lena_unpub.jpg')


def main():
    test()


if __name__ == '__main__':
    main()
