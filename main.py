import torch
import torch.nn as nn
from unet_128 import UNet
import numpy as np
from PIL import Image as im
import scipy.io as sc

data = sc.loadmat('data.mat')
images_dataset = np.array(data['xn'])
array = images_dataset[:,:,1,1]
array = abs((array.real))*10**4
# print(array)

image_tensor = torch.from_numpy(images_dataset)

# image = im.fromarray(array)
# print(image.size)

model = UNet()
print(model(image_tensor))
