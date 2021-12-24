import torch
import os
import evaluate.models as models
from evaluate.Detection_nets import FasterRCNN
from evaluate.utils import non_max_suppression

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from albumentations import RandomBrightnessContrast, RandomGamma, CLAHE, Compose
import warnings
warnings.filterwarnings("ignore")

devicetype = "cuda" if torch.cuda.is_available() else "cpu"
weights = './evaluate/RetinaNet.pth'
image_size = (800,800)
depth = 101
max_repeat = 10
# img_path = "IDRiD_103.jpg"
threshold = 0.008

def find_center(img_path):
    device = torch.device(devicetype)
    model_state_dict = torch.load(weights)
    # image_mean = (0.46737722, 0.24098666, 0.10314517)
    # image_std = (0.04019115, 0.024475794, 0.02510888)

    # model = FasterRCNN(image_mean, image_std, 3)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    if depth == 50:
        model = models.resnet50(num_classes=3, pretrained=True)
    elif depth == 101:
        model = models.resnet101(num_classes=3, pretrained=True)
    elif depth == 152:
        model = models.resnet152(num_classes=3, pretrained=True)
    
    model.load_state_dict(model_state_dict)
    model.to(device)

    image = Image.open(img_path)
    image = np.array(image)
    image = np.uint8(image)
    light = Compose([
        #RandomBrightnessContrast(p=1),    
        #RandomGamma(p=1),    
        CLAHE(p=1),    
        ], p=1)
    image = light(image=image)['image']
    image = Image.fromarray(image)

    init_shape = np.array(list(image.size))
    scale = transforms.Resize(image_size)

    # scale_factor = np.ones(2)
    to_tensor = transforms.ToTensor()
    composed = transforms.Compose([scale, to_tensor])
    image = composed(image)

    model.eval()
    with torch.no_grad(): 
        scores, labels, boxes = model(image.unsqueeze(0).cuda())
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        boxes  = boxes.cpu().numpy()
        # prediction = model([image.to(device)])
        # boxes = prediction[0]['boxes'].cpu().numpy()
        # scores = prediction[0]['scores'].cpu().numpy()
        # labels = prediction[0]['labels'].cpu().numpy()
    
    OD_predicted_box = boxes[0]
    labels = labels[1:]
    scores = scores[1:]
    Fovea_boxes = boxes[1:]

    if len(Fovea_boxes) == 0:
        print("Repeat "+img_path[17:]+" for 1 time")
        try:
            x = find_center(img_path)
        except RuntimeError:
            print(img_path[17:]+" 寄了。")
            pass
        else:
            return x
    if len(Fovea_boxes)>0:   
        kept_idx = list(non_max_suppression(Fovea_boxes, scores, threshold))
        Fovea_boxes = [list(boxes[1:][i]) for i in range(len(boxes[1:])) ]  
        if len(kept_idx)==0:
            Fovea_predicted_box = boxes[1]
        else:
            Fovea_predicted_box = Fovea_boxes[kept_idx[0]]
    else :
        print("Fovea boxes empty for img ")
        Fovea_predicted_box = boxes[0]
    OD_center = [(OD_predicted_box[0]+OD_predicted_box[2])/2,(OD_predicted_box[1]+OD_predicted_box[3])/2]
    Fovea_center = [(Fovea_predicted_box[0]+Fovea_predicted_box[2])/2,(Fovea_predicted_box[1]+Fovea_predicted_box[3])/2]
    print('OD: ',OD_center,'     ','Fo: ',Fovea_center)
    return OD_center+Fovea_center

# print(find_center(img_path))