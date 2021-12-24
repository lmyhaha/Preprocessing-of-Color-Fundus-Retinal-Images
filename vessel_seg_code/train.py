import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from Dataset import DRIVEDatasets, DRIVEDatasets_for_test
from apex import amp

from model.Unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The device is:", device)

# parameters
in_channels = 3
n_classes = 1

learning_rate = 0.0005

dataset_path = "Data_trans/training"
test_path = "Data_trans/test"
batch_size = 2
start_epochs = 0
epochs = 400
version = 1

img_size = 608

# model_name = "unet.pkl"

model = UNet(in_channels=in_channels, n_classes=n_classes, padding=True, up_mode='upconv').to(device)
# model = torch.load(f'trained_model_v{version-1}/unet_v{version-1}_1000.pkl').to(device)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0)

dataloader = DataLoader(DRIVEDatasets(dataset_path, img_size), batch_size=batch_size, shuffle=True)

criteria = nn.BCELoss()

amp.register_float_function(torch, 'sigmoid')
model, optimizer = amp.initialize(model, optim, opt_level="O1")

for e in range(start_epochs, epochs+1):
    running_loss = 0
    i = 0
    for X, target in dataloader:
        X = X.to(device)  # [N, 1, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)

        prediction = model(X)  # [N, 2, H, W]

        loss = criteria(prediction, target.float())
        optim.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optim.step()

        running_loss += loss.item()
        i += 1

    print(f"Epoch {e}, loss: ", running_loss/(i+1))

    if e % 100 == 0:
        # 保存
        torch.save(model, f"trained_model_v{version}/unet_v{version}_{e}.pkl")
        test_size = 1
        # test
        dataloader_test = DataLoader(DRIVEDatasets_for_test(test_path, img_size), batch_size=test_size, shuffle=False)

        i = 0
        running_loss = 0
        for X, target in dataloader_test:
            X = X.to(device)  # [N, 1, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)

            prediction = model(X)  # [N, 2, H, W]

            loss = criteria(prediction, target.float())
            running_loss += loss.item()
            for j in range(test_size):
                r = prediction[j, 0].cpu().detach().numpy()
                r[r > 0.5] = 255
                r[r <= 0.5] = 0
                cv2.imwrite(f"test_result_v{version}/epoch_{e}_{i}.jpg", r)
                i += 1
        print(f"##############################Testing, loss: ", running_loss / (i+1))
