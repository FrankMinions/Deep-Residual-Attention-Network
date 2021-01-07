import torch
import numpy as np
from torch.autograd import Variable
from metrics import ssim, psnr
import warnings

warnings.filterwarnings("ignore")

def test(net, loader_test, device):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, data in enumerate(loader_test):
        inputs = data['input']; targets = data['target']
        inputs = inputs.type(torch.FloatTensor).to(device); targets = targets.type(torch.FloatTensor).to(device)
        inputs_v, targets_v = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)
        pre = net(inputs_v)

        ssim1 = ssim(pre, targets_v).item()
        psnr1 = psnr(pre, targets_v)
        ssims.append(ssim1)
        psnrs.append(psnr1)

    return np.mean(ssims), np.mean(psnrs)