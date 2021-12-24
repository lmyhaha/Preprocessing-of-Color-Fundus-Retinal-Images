import torch

from color_normalization import *


def img2tensor(pic: np.ndarray, device):
    """
    将opencv图片转换到cuda tensor， 经过转换后的图像可以直接输入训练后的模型
    :param pic: 图片
    :param device: 运行设备 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    :return: tensor
    """
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic_t = torch.zeros((1, pic.shape[2], pic.shape[0], pic.shape[1])).to(device)
    pic = torch.from_numpy(pic).to(device)
    pic_t[0, 0, :, :] = pic[:, :, 0]
    pic_t[0, 1, :, :] = pic[:, :, 1]
    pic_t[0, 2, :, :] = pic[:, :, 2]
    return pic_t / 255


def get_seg_img(img: np.ndarray, ref_img: np.ndarray, model, device, dsize:tuple = (608, 608)):
    """
    得到分割后的血管图像
    :param img: 待分割图像
    :param ref_img: 颜色归一化参考图像
    :param model: 神经网络模型
    :param device: 运行设备 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    :param dsize: 输入模型的图片大小，由于模型未特别针对尺度进行训练，建议以DRIVE数据集图片resize到608x608的尺度为标准进行调整
    :return: 分割后的mask，由于模型进行了下采样和上采样，输出图片的大小可能输入不一样(dsize==(608,608),能保持一样)
    """

    img = norm(ref_img, img)  # 进行颜色归一化，由于训练是基于DRIVE数据集，参考图片应选择DRIVE数据中的图片
    img = cv2.resize(img, dsize)  # 在这个大小下效果比较好  训练过程中未对尺度进行单独训练

    input_img = img2tensor(img, device).to(device)
    prediction = model(input_img)

    seg_img = prediction[0, 0]
    seg_img[seg_img > 0.5] = 255
    seg_img[seg_img <= 0.5] = 0
    return seg_img.cpu().detach().numpy()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The device is:", device)

    # 加载模型
    model = torch.load("unet.pkl").to(device)

    ref_path = "DRIVE/test/images/01_test.tif"
    img_path = "DRIVE/test/images/02_test.tif"

    ref_img = cv2.imread(ref_path)
    img = cv2.imread(img_path)

    seg = get_seg_img(img, ref_img, model, device)

    cv2.imwrite(f"result.jpg", seg)
