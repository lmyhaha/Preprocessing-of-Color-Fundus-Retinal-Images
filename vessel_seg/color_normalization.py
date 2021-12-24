import cv2
import numpy as np


def get(img):
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, _ = img.shape
    num = h * w
    # print(num)
    for i in range(h):
        for j in range(w):
            if img_g[i][j] < 5:
                num = num - 1
    # print(num)

    R_channel, G_channel, B_channel = 0, 0, 0
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])
    R_mean, G_mean, B_mean = R_channel / num, G_channel / num, B_channel / num

    R_channel, G_channel, B_channel = 0, 0, 0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var, G_var, B_var = np.sqrt(R_channel / num), np.sqrt(G_channel / num), np.sqrt(B_channel / num)

    return R_mean, G_mean, B_mean, R_var, G_var, B_var


# src模板图像
def norm(img_s, img):
    src = get(img_s)
    inf = get(img)

    mean = np.array([inf[0], inf[1], inf[2]])
    var = np.array([inf[3], inf[4], inf[5]])
    src_mean = np.array([src[0], src[1], src[2]])
    src_var = np.array([src[3], src[4], src[5]])

    img_2 = (src_var / var) * (img - mean) + src_mean
    img_2[img_2 < 0] = 0
    img_2[img_2 > 255] = 255
    img_2 = np.array(img_2, np.uint8)
    return img_2


if __name__ == '__main__':
    img = cv2.imread('001\\01_test.tif')
    img_g = cv2.imread('001\\01_test.tif', 0)

    for i in range(2, 21):
        tmp = str(i).zfill(2)
        pic = f'001\\{tmp}_test.tif'
        pic_ = cv2.imread(pic)
        pic_g = cv2.imread(pic, 0)

        img_f = norm(img, img_g, pic_, pic_g)
        cv2.imwrite(f'001\\{tmp}_test_normal.tif', img_f)
        cv2.imshow(tmp, img_f)
        cv2.waitKey(0)
