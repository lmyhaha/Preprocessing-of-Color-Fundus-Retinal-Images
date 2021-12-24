import evaluate.inf as inf  
import os
import align.align as align

img_root_dir = "./evaluate/images"
imglist = []
drive_root_dir = "./evaluate/train"
drivelist = []
for root, dirs, files in os.walk(img_root_dir, topdown=False):
    for name in files:
        imglist.append(os.path.join(root, name))

for root, dirs, files in os.walk(drive_root_dir, topdown=False):
    for name in files:
        drivelist.append(os.path.join(root, name))
# for i in range(len(imglist)):   
#     print(imglist[i][18:20])
#     print(drivelist[i][17:19])
for i in range(len(imglist)):
    center = inf.find_center(imglist[i])
    align.align(imglist[i], center)
    align.align(drivelist[i], center)
    print(imglist[i],center)