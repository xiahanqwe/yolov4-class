import imp
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
img = Image.open("img/street.jpg")
# print(img.type)
img.show(title="streetsss")
img = np.array(img)/255
img = Image.fromarray(img)
# print("images = ",np.array(img))
# print("imgs = ",imgs)
# imgs = 
img.save("")