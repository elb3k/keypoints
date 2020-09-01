import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, epochs, batch_size
from model import Keypoints
from dataset import KeypointsDataset, transform

from tqdm import tqdm


def custom_loss(predictions_maps, maps, predictions_offsets_x, offsets_x, predictions_offsets_y, offsets_y):
    
    loss_h = bceLoss(predictions_maps, maps)

    distance_x = predictions_offsets_x[maps==1] - offsets_x[maps==1]
    distance_y = predictions_offsets_y[maps==1] - offsets_y[maps==1]
    distances = torch.sqrt(distance_x * distance_x + distance_y * distance_y)
    zero_distances = Variable(
        torch.zeros(distance_x.shape).cuda() if use_cuda else torch.zeros(distance_x.shape))
    loss_o = smoothL1Loss(distances, zero_distances)
    loss = 4 * loss_h + loss_o
    return loss

def forward(sample_batched, model):
    X = sample_batched[0]
    maps, offsets_x, offsets_y = sample_batched[1]

    maps = Variable(maps.cuda() if use_cuda else maps)
    offsets_x = Variable(offsets_x.cuda() if use_cuda else offsets_x)
    offsets_y = Variable(offsets_y.cuda() if use_cuda else offsets_y)

    # forward
    X = Variable(X.cuda() if use_cuda else X)
    (predictions_maps, predictions_offsets_x, predictions_offsets_y), pred = model.forward(X)
    
    return custom_loss(predictions_maps, maps, predictions_offsets_x, offsets_x, predictions_offsets_y, offsets_y)

def fit(train_data, test_data, model, loss_function, epochs, initial_epoch=0, checkpoint_path='', tensorboard_path=''):

    global optimizer

    writer = SummaryWriter(log_dir=tensorboard_path)

    for epoch in range(initial_epoch, epochs):

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)

        # training 
        train_loss = 0.0
        progressBar = tqdm(enumerate(train_data), desc="Epoch: %03d, loss: %02.3f"%(epoch, 0.0)) 
        for i_batch, sample_batched in progressBar:
            optimizer.zero_grad()
            
            loss = forward(sample_batched, model)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()

            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.data.item()), end='')
            # print('\r', end='')
            progressBar.set_description("Epoch: %03d, loss: %02.3f"%(epoch, loss.data.item()))
        train_loss /= i_batch
        print('train loss:', train_loss)

        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.data.item()

        test_loss /= i_batch
        print('test loss:', test_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", test_loss, epoch)
        
        torch.save(keypoints.module.state_dict(), checkpoint_path + 'model_' + str(epoch)+'.pth')

# dataset
# Train
train_dataset = KeypointsDataset('dataset/keypoints/train.h5',
                           NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, transform=transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)


# Test
test_dataset = KeypointsDataset('dataset/keypoints/val.h5',
                           NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, transform=transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

use_cuda = torch.cuda.is_available()
# use_cuda = False
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# if use_cuda:
#     torch.cuda.set_device(2)

# loss
smoothL1Loss = nn.SmoothL1Loss()
bceLoss = nn.BCELoss()
# model
keypoints = Keypoints(NUM_CLASSES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load("weights/model_19.pth"))
keypoints = nn.DataParallel(keypoints.cuda()) if use_cuda else keypoints




# Learning rate
lr = 0.0001

def adjust_learning_rate(optimzer, epoch):

    new_lr = lr * (0.1) ** (epoch//20)
    for param_group in optimzer.param_groups:
        param_group["lr"] = new_lr

# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=lr)

fit(train_data, test_data, keypoints, custom_loss, initial_epoch=0, epochs=60, checkpoint_path='weights/v1/', tensorboard_path="log/v1")
