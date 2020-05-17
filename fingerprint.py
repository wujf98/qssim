import json
from itertools import product

import numpy as np
from cv2 import cv2


def iter_block(grayscale_image, nh, nw):
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
    grayscale_image = cv2.resize(grayscale_image, (10, 10), interpolation=cv2.INTER_CUBIC)
    grayscale_image = grayscale_image.astype(np.int32)
    start_i, start_j, base_h, base_w = 1, 1, 8, 8
    center = grayscale_image[start_i: start_i + base_h, start_j: start_j + base_w]
    hash_str = ''
    for offset_i, offset_j in product((-1, 0, 1), (-1, 0, 1)):
        if offset_i == 0 and offset_j == 0:
            continue
        offset = grayscale_image[start_i + offset_i: start_i + base_h + offset_i, start_j + offset_j: start_j + base_w + offset_j]
        difference_value_image = center - offset
        hash_str += ''.join(map(lambda x: '1' if x > 0 else '0', difference_value_image.ravel()))
    return hash_str


def blocked_fingerprint(bgr_image, nh=4, nw=4, fingerprint_function_list=[np.mean, np.var, difference_value_hash]):
    grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    res = list()
    for block in iter_block(grayscale_image, nh, nw):
        block_fingerprint = list(func(block) for func in fingerprint_function_list)
        res.append(block_fingerprint)
    return res


def test():
    pass


def main():
    test()


if __name__ == '__main__':
    main()
