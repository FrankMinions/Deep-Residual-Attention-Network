import os
import torch
import warnings
import numpy as np
import torchvision
from torchvision import transforms
from draa import main
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataLoader import SalObjDataset, ToTensorLab

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

denorm = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
model_path = './saved_models/draa_mse_itr_26600_train_0.052256_max_ssim_0.848528_max_psnr_20.599444.pth'

def test(net, loader_test, device):
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, data in enumerate(loader_test):
        inputs = data['input']; targets = data['target']
        inputs = inputs.type(torch.FloatTensor).to(device); targets = targets.type(torch.FloatTensor).to(device)
        inputs_v, targets_v = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)
        model = torch.load(model_path)
        net.load_state_dict(model)
        pre = net(inputs_v)
        pre = pre.clone().squeeze()
        pre = denorm(pre).clamp(0, 1)
        torchvision.utils.save_image(pre, os.path.join('./test_results/', str(i)+'.jpg'))

    return np.mean(ssims), np.mean(psnrs)

xtes_img_name_list = os.listdir('./test_mosaic/')
ytes_img_name_list = os.listdir('./test_raw')
for idx, img in enumerate(xtes_img_name_list):
    xtes_img_name_list[idx] = os.path.join('./test_mosaic/', xtes_img_name_list[idx])

for idx, img in enumerate(ytes_img_name_list):
    ytes_img_name_list[idx] = os.path.join('./test_raw/', ytes_img_name_list[idx])

test_dataset = SalObjDataset(
    x_img_name_list=xtes_img_name_list,
    y_img_name_list=ytes_img_name_list,
    transform=transforms.Compose([
        ToTensorLab(flag=0)
    ])
)

if __name__ == "__main__":
    loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = main()
    net.to(device).eval()

    test(net, loader_test, device)
