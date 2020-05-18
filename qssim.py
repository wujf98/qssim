import numpy as np


# C1，C2，C3是作除法时引入的常数，作用是避免除零。
K1, K2, L = 0.01, 0.03, 255
C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
C3 = C2 / 2


def qssim(fingerprint_1, fingerprint_2):
    """
    计算两图像对应区域的匹配程度。

    参数：
        fingerprint_1: 图像某一块区域的指纹信息，包括均值、方差和差值哈希字符串。
        fingerprint_2: 同上。
    返回：
        两图像对应区域的匹配程度，为[0, 1]内某数，越接近1，表示两区域匹配程度越高。
    """

    assert len(fingerprint_1) == 3 and len(fingerprint_2) == 3
    
    mean_1, var_1, d_hash_1 = fingerprint_1
    mean_2, var_2, d_hash_2 = fingerprint_2

    l = (2 * mean_1 * mean_2 + C1) / (mean_1 ** 2 + mean_2 ** 2 + C1)
    c = (2 * (var_1 * var_2) ** 0.5 + C2) / (var_1 + var_2 + C2)

    d_hash_1 = np.array(list(map(int, d_hash_1)))
    d_hash_2 = np.array(list(map(int, d_hash_2)))

    # 计算交并比
    i = np.sum(np.logical_and(d_hash_1, d_hash_2))
    u = np.sum(np.logical_or(d_hash_1, d_hash_2))
    s = (i + C3) / (u + C3)

    return l * c * s


def mqssim(fingerprint_list_1, fingerprint_list_2):
    """
    计算两图像所有对应区域的平均匹配程度。

    参数：
        fingerprint_list_1: 图像所有区域指纹信息的列表。
        fingerprint_list_2: 同上。

    返回：
        两图像所有对于区域的平均匹配程度，为[0, 1]内某数，越接近1，表示两图像匹配程度越高。
    """

    assert len(fingerprint_list_1) == len(fingerprint_list_2)

    s = sum(qssim(x[0], x[1]) for x in zip(fingerprint_list_1, fingerprint_list_2))
    m = s / len(fingerprint_list_1)
    
    return m


def test():
    fingerprint_list_1 = [
        (47.23886705154977, 597.3311892663193, '0010010000100100001001000010001000100110001001010010010000100100'),
        (84.15833538058388, 1207.2185332729348, '1111011001110110011001100111011010000010010101100011000100110100'),
        (83.58799492281865, 1907.39929636455, '0101010101010101010101010101010110010101010101011010100111010000'),
        (58.56914691069893, 284.8424160187482, '1001011010010110110101101001011010010010100110101001101010011010'),
    ]
    fingerprint_list_2 = [
        (46.86687939780723, 604.1213887072681, '0010010000100100001001000010001000100110001001010010010000100100'),
        (83.78947799050891, 1222.7433257134333, '1011011001110110011101100111011010100010010101100011000100110101'),
        (83.20475372279496, 1913.912297807945, '0101010101010101010101010101010110010101010101011010100111010000'),
        (58.20092456226477, 284.097421825597, '0001011011010110110101101001011010010010100110101001101010011010'),
    ]
    v = mqssim(fingerprint_list_1, fingerprint_list_2)
    print(v)


def main():
    test()


if __name__ == '__main__':
    main()
