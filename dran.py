import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel attention network
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel//8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//8, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_ca_avg = self.ca(y_avg)
        y_ca_max = self.ca(y_max)
        y = self.sigmoid(y_ca_avg+y_ca_max)

        return x * y

# Residual block (Conv+BN+ReLU pair)
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu(self.bn(self.conv(hx)))
        return xout

# Residual block (Conv+BN pair)
class ResBlockLast(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlockLast, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        hx = x
        xout = self.bn(self.conv(hx))
        return xout

# Encoder and decoder convolutional layers
class ENDECONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate):
        super(ENDECONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu(self.bn(self.conv(hx)))
        return xout

# Upsampling
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src

class DRAN(nn.Module):
    def __init__(self):
        super(DRAN, self).__init__()

        # Downsampling
        self.max_pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avg_pool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.encoconv1 = ENDECONV(3, 32, dirate=1)
        self.encoconv2 = ENDECONV(32, 64, dirate=1)
        self.encoconv3 = ENDECONV(64, 128, dirate=1)
        self.encoconv4 = ENDECONV(128, 256, dirate=1)
        self.encoconv5 = ENDECONV(256, 512, dirate=1)
        self.encoconv6 = ENDECONV(512, 1024, dirate=1)
        self.encoconv7 = ENDECONV(1024, 1024, dirate=1)

        self.encoconv1_ = ENDECONV(32*2, 32, dirate=1)
        self.encoconv2_ = ENDECONV(64*2, 64, dirate=1)
        self.encoconv3_ = ENDECONV(128*2, 128, dirate=1)
        self.encoconv4_ = ENDECONV(256*2, 256, dirate=1)
        self.encoconv5_ = ENDECONV(512*2, 512, dirate=1)
        self.encoconv6_ = ENDECONV(1024*2, 1024, dirate=1)

        self.decoconv0 = ENDECONV(1024*2, 512, dirate=1)
        self.decoconv1 = ENDECONV(512*2, 256, dirate=1)
        self.decoconv2 = ENDECONV(256*2, 128, dirate=1)
        self.decoconv3 = ENDECONV(128*2, 64, dirate=1)
        self.decoconv4 = ENDECONV(64*2, 32, dirate=1)
        self.decoconv5 = ENDECONV(32*2, 3, dirate=1)

        self.resblock1 = ResBlock(32, 32)
        self.resblock2 = ResBlock(64, 64)
        self.resblock3 = ResBlock(128, 128)
        self.resblock4 = ResBlock(256, 256)
        self.resblock5 = ResBlock(512, 512)
        self.resblock6 = ResBlock(1024, 1024)

        self.resblocklast1 = ResBlockLast(32, 32)
        self.resblocklast2 = ResBlockLast(64, 64)
        self.resblocklast3 = ResBlockLast(128, 128)
        self.resblocklast4 = ResBlockLast(256, 256)
        self.resblocklast5 = ResBlockLast(512, 512)
        self.resblocklast6 = ResBlockLast(1024, 1024)

        self.calayer1 = CALayer(32)
        self.calayer2 = CALayer(64)
        self.calayer3 = CALayer(128)
        self.calayer4 = CALayer(256)
        self.calayer5 = CALayer(512)
        self.calayer6 = CALayer(1024)

        self.post = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        hx = x

        hx1 = self.encoconv1(hx)     # 3->32
        hx1r1 = self.resblock1(hx1)
        hx1r2 = self.resblock1(hx1r1)
        hx1r3 = self.resblocklast1(hx1r2)
        hx1r1_ = self.resblock1(hx1)
        hx1r2_ = self.resblock1(hx1r1_)
        hx1r3_ = self.resblocklast1(hx1r2_)
        hx11 = nn.ReLU(inplace=True)(hx1r3 + hx1)
        hx1_ = nn.ReLU(inplace=True)(hx1r3_ * hx1)
        hx1 = torch.cat((hx11, hx1_), 1)
        hx1 = self.encoconv1_(hx1)

        hx_max = self.max_pool(hx1)
        hx_avg = self.avg_pool(hx1)
        hx = hx_avg + hx_max
        hx2 = self.encoconv2(hx)        # 32->64
        hx2r1 = self.resblock2(hx2)
        hx2r2 = self.resblock2(hx2r1)
        hx2r3 = self.resblocklast2(hx2r2)
        hx2r1_ = self.resblock2(hx2)
        hx2r2_ = self.resblock2(hx2r1_)
        hx2r3_ = self.resblocklast2(hx2r2_)
        hx22 = nn.ReLU(inplace=True)(hx2r3 + hx2)
        hx2_ = nn.ReLU(inplace=True)(hx2r3_ * hx2)
        hx2 = torch.cat((hx22, hx2_), 1)
        hx2 = self.encoconv2_(hx2)

        hx_max = self.max_pool(hx2)
        hx_avg = self.avg_pool(hx2)
        hx = hx_avg + hx_max
        hx3 = self.encoconv3(hx)          # 64->128
        hx3r1 = self.resblock3(hx3)
        hx3r2 = self.resblock3(hx3r1)
        hx3r3 = self.resblocklast3(hx3r2)
        hx3r1_ = self.resblock3(hx3)
        hx3r2_ = self.resblock3(hx3r1_)
        hx3r3_ = self.resblocklast3(hx3r2_)
        hx33 = nn.ReLU(inplace=True)(hx3r3 + hx3)
        hx3_ = nn.ReLU(inplace=True)(hx3r3_ * hx3)
        hx3 = torch.cat((hx33, hx3_), 1)
        hx3 = self.encoconv3_(hx3)

        hx_max = self.max_pool(hx3)
        hx_avg = self.avg_pool(hx3)
        hx = hx_max + hx_avg
        hx4 = self.encoconv4(hx)             # 128->256
        hx4r1 = self.resblock4(hx4)
        hx4r2 = self.resblock4(hx4r1)
        hx4r3 = self.resblocklast4(hx4r2)
        hx4r1_ = self.resblock4(hx4)
        hx4r2_ = self.resblock4(hx4r1_)
        hx4r3_ = self.resblocklast4(hx4r2_)
        hx44 = nn.ReLU(inplace=True)(hx4r3 + hx4)
        hx4_ = nn.ReLU(inplace=True)(hx4r3_ * hx4)
        hx4 = torch.cat((hx44, hx4_), 1)
        hx4 = self.encoconv4_(hx4)

        hx_max = self.max_pool(hx4)
        hx_avg = self.avg_pool(hx4)
        hx = hx_avg + hx_max
        hx5 = self.encoconv5(hx)            # 256->512
        hx5r1 = self.resblock5(hx5)
        hx5r2 = self.resblock5(hx5r1)
        hx5r3 = self.resblocklast5(hx5r2)
        hx5r1_ = self.resblock5(hx5)
        hx5r2_ = self.resblock5(hx5r1_)
        hx5r3_ = self.resblocklast5(hx5r2_)
        hx55 = nn.ReLU(inplace=True)(hx5r3 + hx5)
        hx5_ = nn.ReLU(inplace=True)(hx5r3_ * hx5)
        hx5 = torch.cat((hx55, hx5_), 1)
        hx5 = self.encoconv5_(hx5)

        hx_max = self.max_pool(hx5)
        hx_avg = self.avg_pool(hx5)
        hx = hx_avg + hx_max
        hx6 = self.encoconv6(hx)            # 512->1024
        hx6r1 = self.resblock6(hx6)
        hx6r2 = self.resblock6(hx6r1)
        hx6r3 = self.resblocklast6(hx6r2)
        hx6r1_ = self.resblock6(hx6)
        hx6r2_ = self.resblock6(hx6r1_)
        hx6r3_ = self.resblocklast6(hx6r2_)
        hx66 = nn.ReLU(inplace=True)(hx6r3 + hx6)
        hx6_ = nn.ReLU(inplace=True)(hx6r3_ * hx6)
        hx6 = torch.cat((hx66, hx6_), 1)
        hx6 = self.encoconv6_(hx6)

        hx7 = self.encoconv7(hx6)                   # 1024->1024

        hx7up = _upsample_like(hx7, hx6)
        hx7d = self.decoconv0(torch.cat((hx7up, self.calayer6(hx6)), 1))  # 1024->512

        hx6up = _upsample_like(hx7d, hx5)
        hx6d = self.decoconv1(torch.cat((hx6up, self.calayer5(hx5)), 1))  # 512->256

        hx5up = _upsample_like(hx6d, hx4)
        hx5d = self.decoconv2(torch.cat((hx5up, self.calayer4(hx4)), 1))  # 256->128

        hx4up = _upsample_like(hx5d, hx3)
        hx4d = self.decoconv3(torch.cat((hx4up, self.calayer3(hx3)), 1))  # 128->64

        hx3up = _upsample_like(hx4d, hx2)
        hx3d = self.decoconv4(torch.cat((hx3up, self.calayer2(hx2)), 1))  # 64->32

        hx2up = _upsample_like(hx3d, hx1)
        hx2d = self.decoconv5(torch.cat((hx2up, self.calayer1(hx1)), 1))  # 32->3

        hx1d = self.post(hx2d)

        return hx1d + x

def main():
    dran = DRAN()
    return dran




