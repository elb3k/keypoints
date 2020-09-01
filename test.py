import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import cv2
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, RADIUS, epochs, batch_size
from model import Keypoints
from dataset import KeypointsDataset, transform

from tqdm import tqdm

use_cuda = torch.cuda.is_available()

# model
model = Keypoints(NUM_CLASSES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_small_height=IMG_SMALL_HEIGHT, img_small_width=IMG_SMALL_WIDTH)
# model = model.cuda()
model.load_state_dict(torch.load("sample_weights.pth"))
model = model.cuda() if use_cuda else model



# Load image
img = cv2.imread("sample/3.jpg")

# Predict
(maps_array, offsets_x_array, offsets_y_array), keypoints = model.predict(img)

print(keypoints)

keypoints = keypoints.cpu()

# Drawing function
def draw_keypoints(img, keypoints):

    height, width = img.shape[:2]

    for y, x in keypoints:
        
        x = x * width / IMG_SMALL_WIDTH
        y = y * height / IMG_SMALL_HEIGHT
        img = cv2.circle( img, (x, y), 5, (0, 0, 255), -1)

    return img


img = draw_keypoints(img, keypoints)

cv2.imwrite("result.png", img)
