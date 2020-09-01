import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

pre_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) ])

class Keypoints(nn.Module):
    def __init__(self, num_classes, img_height=256, img_width=186, img_small_height=88, img_small_width=72, resnet=101):
        super(Keypoints, self).__init__()
        
        self.num_classes = num_classes
        self.num_outputs = num_classes * 3
        self.img_height = img_height
        self.img_width = img_width

        self.img_small_height = img_small_height
        self.img_small_width = img_small_width


        if resnet == 18:
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.conv1by1 = nn.Conv2d(512, self.num_outputs, (1,1))
        elif resnet == 101:
            self.resnet = torchvision.models.resnet101(pretrained=True)
            self.conv1by1 = nn.Conv2d(2048, self.num_outputs, (1,1))

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet
            
        self.conv_transpose = nn.ConvTranspose2d(self.num_outputs, self.num_outputs, kernel_size=32, stride=8)
        self.sigmoid = torch.nn.Sigmoid()


        #region For prediction

        self.offset_x_ij = torch.arange(0, self.img_small_width) \
            .repeat(self.img_small_height).view(1,1,self.img_small_height, self.img_small_width)
        self.offset_y_ij = torch.arange(0, self.img_small_height) \
            .repeat(self.img_small_width).view(self.img_small_width, self.img_small_height).t().contiguous() \
            .view(1,1,self.img_small_height, self.img_small_width)
        
        self.offset_x_ij = self.offset_x_ij.cuda()
        self.offset_y_ij = self.offset_y_ij.cuda()
    
        self.offset_x_add = (0 - self.offset_x_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        self.offset_y_add = (0 - self.offset_y_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        
        self.offset_x_ij = (self.offset_x_ij + self.offset_x_add) * self.img_width / self.img_small_width
        self.offset_y_ij = (self.offset_y_ij + self.offset_y_add) * self.img_height/ self.img_small_height


        #endregion
        
    def forward(self, x):

        x = self.resnet(x)
        x = self.conv1by1(x)
        x = self.conv_transpose(x)
        output = nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear', align_corners=True)(x)

        maps = self.sigmoid(output[:,:self.num_classes, :, :])
        offsets_x = output[:, self.num_classes:2*self.num_classes, :, :]
        offsets_y = output[:, 2*self.num_classes:3*self.num_classes, :, :]
        
        maps_pred = self.sigmoid(x[:,:self.num_classes, :, :])
        offsets_x_pred = x[:, self.num_classes:2*self.num_classes, :, :]
        offsets_y_pred = x[:, 2*self.num_classes:3*self.num_classes, :, :]

        return (maps, offsets_x, offsets_y), (maps_pred, offsets_x_pred, offsets_y_pred)


    def predict(self, img):

        # Convert cv2 image to tensor and apply preprocessing
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = pre_process(Image.fromarray(img)).cuda()

        # Forward
        result, (maps_pred, offsets_x_pred, offsets_y_pred) = self.forward( Variable(img.unsqueeze(0)) )
        maps_pred = maps_pred.data[0]
        offsets_x_pred = offsets_x_pred.data[0]
        offsets_y_pred = offsets_y_pred.data[0]


        # Keypoint coordinates
        keypoints = torch.zeros(self.num_classes, 2, 1)
        keypoints = keypoints.type(torch.cuda.LongTensor)

        # For every keypoint or classs
        for i in range(self.num_classes):
            
            offsets_x_i = self.offset_x_ij + offsets_x_pred[i]
            offsets_y_i = self.offset_y_ij + offsets_y_pred[i]

            distances_i = torch.sqrt(offsets_x_i * offsets_x_i + offsets_y_i * offsets_y_i)

            distances_i[distances_i > 1] = 1
            distances_i = 1 - distances_i

            score_i = (distances_i * maps_pred[i]).sum(3).sum(2)

            v1, index_y = score_i.max(0)
            v2, index_x = v1.max(0)

            keypoints[i][0] = index_y[index_x]
            keypoints[i][1] = index_x

        # To shape num_classes x 2
        keypoints = keypoints.view(self.num_classes, 2)

        maps_array = result[0][0]
        offsets_x_array = result[1][0]
        offsets_y_array = result[2][0]

        return (maps_array, offsets_x_array, offsets_y_array), keypoints        
