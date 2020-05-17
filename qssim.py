import numpy as np


K1, K2, L = 0.01, 0.03, 255
C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
C3 = C2 / 2


def qssim(fingerprint_1, fingerprint_2):
    assert len(fingerprint_1) == 3 and len(fingerprint_2) == 3
    
    mean_1, var_1, d_hash_1 = fingerprint_1
    mean_2, var_2, d_hash_2 = fingerprint_2

    l = (2 * mean_1 * mean_2 + C1) / (mean_1 ** 2 + mean_2 ** 2 + C1)
    c = (2 * var_1 * var_2 + C2) / (var_1 ** 2 + var_2 ** 2 + C2)

    d_hash_1 = np.array(list(map(int, d_hash_1)))
    d_hash_2 = np.array(list(map(int, d_hash_2)))

    i = np.sum(np.logical_and(d_hash_1, d_hash_2))
    u = np.sum(np.logical_or(d_hash_1, d_hash_2))
    s = (i + C3) / (u + C3)

    return l * c * s


def mqssim(fingerprint_list_1, fingerprint_list_2):
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
        (57.41806391516194, 1182.9031460366145, '0010011000100100001001000010010000100100011001000110110101100101'),
        (90.12158160340663, 3088.5597966919145, '1000011010100101101010001101100010011001101000111000101000101110'),
        (85.66591686115547, 3391.5024505085007, '1110000101000011010010110100101111001001100110011001100110010101'),
        (59.06707560496253, 563.8482191001267, '1001101110011010100111101001101101011110100101101010111010101100'),
        (47.507385456332145, 855.876402073796, '0100010001001100010110100100100011001001110001011110001101101010'),
        (104.50320599434959, 4555.408268802792, '0111001000100100011001100010011000100110010100100101101001000010'),
        (73.17742343282971, 2891.54065426291, '1001010111000101110011101101011010011010100111010001011000100010'),
        (46.16787249723621, 1153.903058844324, '1010010010011001111010011010101111011011110111010001101010001001'),
        (44.38443534200622, 721.1998106173828, '0110111010101010011100001010010000011000100111101010001111101011'),
        (97.37030252822778, 4544.604700207118, '1000001000010010110100111000001010011010100010010100011001011010'),
        (77.30385114956636, 2251.1572485052534, '0011011000110001001101000111010100100101110100101101010001110110'),
        (59.43600781377843, 1132.2980977672441, '0101000011001110011001010110011101100101111101110111010110011011'),
    ]
    fingerprint_list_2 = [
        (46.86687939780723, 604.1213887072681, '0010010000100100001001000010001000100110001001010010010000100100'),
        (83.78947799050891, 1222.7433257134333, '1011011001110110011101100111011010100010010101100011000100110101'),
        (83.20475372279496, 1913.912297807945, '0101010101010101010101010101010110010101010101011010100111010000'),
        (58.20092456226477, 284.097421825597, '0001011011010110110101101001011010010010100110101001101010011010'),
        (57.042096219931274, 1197.1314385120968, '0010011000100100001001000010010000100100011001010110110101100101'),
        (89.76657666503027, 3125.6564201086285, '0000011010100101101010001101100000011001101000111000101010101110'),
        (85.25990836197022, 3412.014430952165, '1110000101000011010010010100100111001001100110011001100110010101'),
        (58.68413516609393, 567.650163950966, '1001101110011010100111101101101101011110101101101110111110101100'),
        (47.1219440353461, 868.4143720021042, '0100110001001000010110100100101011001001110001011100001101001010'),
        (104.12689412534773, 4598.30069545909, '0111001000110100011101100010011000100110010100100101101001000010'),
        (72.79274259531992, 2931.645941439062, '1001011011000101110011101100011010011010100111010001011000100010'),
        (45.7662166584847, 1166.5418025575432, '1011010010011001111010011010101111001011110111010001101010101010'),
        (44.00552282768778, 727.3445857641245, '0110111100110010011100001010010000100000100111101010001111101011'),
        (96.99150711831125, 4559.609844414898, '1100001000010010110100111000001010011010110010110100011001011010'),
        (76.92866961217477, 2263.1043376016937, '0011011000100001001101000111010100100101110100101101010001110110'),
        (59.0624038618884, 1149.9524549643709, '0101000001001110011001010110111101100101111101110111010110011011'),
    ]
    print(mqssim(fingerprint_list_1, fingerprint_list_2))


def main():
    test()


if __name__ == '__main__':
    main()
