from __future__ import print_function, division
import torch
from skimage import io, transform, color
import numpy as np
from torch.utils.data import Dataset

# ==========================dataset load==========================
class RescaleT(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		x, y = sample['input'], sample['target']

		h, w = x.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# resize the image to (new_h, new_w)
		xt = transform.resize(x, (new_h, new_w), mode='constant')
		yt = transform.resize(y, (new_h, new_w), mode='constant')

		return {'input': xt, 'target': yt}


def TensorOpt(flag, image):
	if flag == 2:  # with rgb and Lab colors
		tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
		tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
		if image.shape[2] == 1:
			tmpImgt[:, :, 0] = image[:, :, 0]
			tmpImgt[:, :, 1] = image[:, :, 0]
			tmpImgt[:, :, 2] = image[:, :, 0]
		else:
			tmpImgt = image
		tmpImgtl = color.rgb2lab(tmpImgt)

		# nomalize image to range [0,1]
		tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
					np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
		tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
					np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
		tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
					np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
		tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
					np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
		tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
					np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
		tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
					np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

		tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
		tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
		tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
		tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
		tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
		tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

	elif flag == 1:  # with Lab color
		tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

		if image.shape[2] == 1:
			tmpImg[:, :, 0] = image[:, :, 0]
			tmpImg[:, :, 1] = image[:, :, 0]
			tmpImg[:, :, 2] = image[:, :, 0]
		else:
			tmpImg = image

		tmpImg = color.rgb2lab(tmpImg)

		tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
					np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
		tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
					np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
		tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
					np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

		tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
		tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
		tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

	else:  # with rgb color
		tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
		image = image / np.max(image)
		if image.shape[2] == 1:
			tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
		else:
			tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
			tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
			tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

	# change the r,g,b to b,r,g
	tmpImg = tmpImg.transpose((2, 0, 1))
	return tmpImg

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self, flag=0):
		self.flag = flag

	def __call__(self, sample):

		x, y = sample['input'], sample['target']

		# change the color space and normalize image
		x = TensorOpt(self.flag, x)
		y = TensorOpt(self.flag, y)

		return {'input': torch.from_numpy(x), 'target': torch.from_numpy(y)}

class SalObjDataset(Dataset):
	def __init__(self, x_img_name_list, y_img_name_list, transform=None):

		self.x_image_name_list = x_img_name_list
		self.y_image_name_list = y_img_name_list
		self.transform = transform

	def __len__(self):
		return len(self.x_image_name_list)

	def __getitem__(self, idx):

		x = io.imread(self.x_image_name_list[idx])
		y = io.imread(self.y_image_name_list[idx])
		# imname = self.x_image_name_list[idx]

		if 2 == len(x.shape):
			x = x[:, :, np.newaxis]
		if 2 == len(y.shape):
			y = y[:, :, np.newaxis]

		sample = {'input': x, 'target': y}

		if self.transform:
			sample = self.transform(sample)

		return sample

