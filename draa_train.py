import os
import torch
from draa import main
from torch.autograd import Variable
from dataLoader import SalObjDataset, ToTensorLab
from draa_test import test
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_dir = './saved/'
model_name = 'draa'

epoch_num = 150
batch_size_train = 20
train_num = 0

xtra_img_name_list = os.listdir('./train_mosaic/')
ytra_img_name_list = os.listdir('./train_raw/')
for idx, img in enumerate(xtra_img_name_list):
    xtra_img_name_list[idx] = os.path.join('./train_mosaic/', xtra_img_name_list[idx])
for idx, img in enumerate(ytra_img_name_list):
    ytra_img_name_list[idx] = os.path.join('./train_raw/', ytra_img_name_list[idx])

xtes_img_name_list = os.listdir('./test_mosaic/')
ytes_img_name_list = os.listdir('./test_raw')
for idx, img in enumerate(xtes_img_name_list):
    xtes_img_name_list[idx] = os.path.join('./test_mosaic/', xtes_img_name_list[idx])
for idx, img in enumerate(ytes_img_name_list):
    ytes_img_name_list[idx] = os.path.join('./test_raw/', ytes_img_name_list[idx])

print("---")
print("face mosaic input images: ", len(xtra_img_name_list))
print("face mosaic target images: ", len(ytra_img_name_list))
print("---")

train_dataset = SalObjDataset(
    x_img_name_list=xtra_img_name_list,
    y_img_name_list=ytra_img_name_list,
    transform=transforms.Compose([
        ToTensorLab(flag=0)
    ])
)

test_dataset = SalObjDataset(
    x_img_name_list=xtes_img_name_list,
    y_img_name_list=ytes_img_name_list,
    transform=transforms.Compose([
        ToTensorLab(flag=0)
    ])
)

loader_train = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

net = main()
net.to(device).eval()

print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
cosinelr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=135000, eta_min=0)

print("---start training...")
max_ssim = 0.0
max_psnr = 0.0
ite_num = 0
running_loss = 0.0
save_frq = 200
psnrs = []
ssims = []
loss = nn.MSELoss()

for epoch in range(0, epoch_num):

    net.train()

    for i, data in enumerate(loader_train):
        ite_num = ite_num + 1

        x_train, y_train = data['input'], data['target']

        x_train = x_train.type(torch.FloatTensor)
        y_train = y_train.type(torch.FloatTensor)

        if torch.cuda.is_available():
            x_train_v, y_train_v = Variable(x_train.cuda(), requires_grad=False), Variable(y_train.cuda(), requires_grad=False)
        else:
            x_train_v, y_train_v = Variable(x_train, requires_grad=False), Variable(y_train, requires_grad=False)

        output = net(x_train_v)
        mse_loss = loss(output, y_train_v)

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        cosinelr.step()

        running_loss = mse_loss

        del output, mse_loss
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f"% (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss))

        if ite_num % save_frq == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, device)

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save(net.state_dict(), model_dir + model_name + "_mse_itr_%d_train_%3f_max_ssim_%4f_max_psnr_%4f.pth" % (ite_num, running_loss, max_ssim, max_psnr))


