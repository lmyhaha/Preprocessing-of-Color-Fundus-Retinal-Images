import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageSequence


def align(imgurl, center):
    preimg = Image.open(imgurl)
    # out = preimg.resize((800, 800), Image.ANTIALIAS)
    preimg.save(r"./tmp.jpg")
    img = cv.imread(r"./tmp.jpg")
    x = img.shape[1]
    y = img.shape[0]
    # x1, y1, x2, y2 = 97, 376, 404, 400
    [x1, y1, x2, y2] = center
    x1 = x1 / 800 * x
    x2 = x2 / 800 * x
    y1 = y1 / 800 * y
    y2 = y2 / 800 * y
    # fs = 200*(np.sqrt((y1 - y2)**2+(x1 - x2)**2)/300)
    fs = (y / 4) * (np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2) / (5 * x / 16))
    y3 = (y1+y2)/2+fs*(x1-x2)/np.sqrt((y1 - y2)**2+(x1 - x2)**2)
    x3 = (x1+x2)/2-fs*(y1-y2)/np.sqrt((y1 - y2)**2+(x1 - x2)**2)
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
    # pts2 = np.float32([[100, 400], [400, 400], [250, 200]])
    pts2 = np.float32([[3 * x / 16, y / 2], [x / 2, y / 2], [11 * x / 32, y / 4]])
    # cv.line(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(0, 255, 0), thickness=10)
    # cv.line(img, pt1=(int(x1), int(y1)), pt2=(int(x3), int(y3)), color=(0, 255, 0), thickness=10)
    # cv.line(img, pt1=(int(x3), int(y3)), pt2=(int(x2), int(y2)), color=(0, 255, 0), thickness=10)
    # print(fs)
    # print(300 / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    # print(np.sqrt(150**2+200**2) / np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))
    # print(np.sqrt(150**2+200**2) / np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
    M = cv.getAffineTransform(pts1, pts2)
    res = cv.warpAffine(img, M, (x, y))
    cv.imwrite(r"./img/tmp.jpg", img)

    cv.imwrite(r"./img/"+imgurl[17:], res) # 这里的17和main.py里的img_root_dir对应！换数据集的话要修改这里和img_root_dir的长度相等！



