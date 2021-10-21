import os
import cv2
import torch
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn as nn
import openslide
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from skimage import measure,data,color

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path = "DATA_PATH"
df_ori = pd.read_excel("XLSX_PATH", encoding='gbk', header=1, sheet_name=1)
val_id = df_ori.iloc[:, 5][df_ori.iloc[:, 2] == "TEST"]

width = 416
height = 416
average_area = 900
tps_ratio = {}


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class Unet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input)  # 64,128,128
        e2 = self.layer2(e1)  # 64,64,64
        e3 = self.layer3(e2)  # 128,32,32
        e4 = self.layer4(e3)  # 256,16,16
        f = self.layer5(e4)  # 512,8,8
        d4 = self.decode4(f, e4)  # 256,16,16
        d3 = self.decode3(d4, e3)  # 256,32,32
        d2 = self.decode2(d3, e2)  # 128,64,64
        d1 = self.decode1(d2, e1)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256
        out = F.sigmoid(out)
        return out


model = Unet(2)
ch = torch.load("CHECKPOINT_PATH")
model.load_state_dict(ch)
model.cuda()
model.eval()

for case_id in val_id:
    folder_path = os.path.join(data_path, case_id)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        try:
            slide = openslide.OpenSlide(img_path)
        except:
            print('read image fail', img_path)
            continue
        w_total, h_total = slide.level_dimensions[0]
        h_times = h_total // height
        w_times = w_total // width
        red_counts = []
        green_counts = []
        green_counts_d = []
        green_counts_e = []
        for item in coord_list:
            i, j = item[0], item[1]
            h_start = i * height
            w_start = j * width
            patch = slide.read_region((w_start, h_start), 0, (height, width))
            patch = np.array(patch)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            mask_final = patch.copy()
            ori_img = patch.copy()
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            save_path = "YOUR_SAVE_PATH"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            cv2.imwrite(save_path, patch)
            # if np.var(patch) < 100:
            #     continue
            img = transforms.ToTensor()(patch)
            img = img.unsqueeze(0)
            inp = Variable(img.cuda(), requires_grad=False)
            res = model(inp)
            out = res[0].cpu().data.numpy()
            mask1 = out[0, ...]
            mask2 = out[1, ...]

            th1,th2 = 0.9,0.05
            mask1[mask1 > th1] = 1
            mask1[mask1 < th1] = 0
            mask2[mask2 > th2] = 1
            mask2[mask1 > th1] = 0
            mask2[mask2 < th2] = 0

            fig, ax = plt.subplots()
            img1 = np.zeros(patch.shape[:2])
            img2 = np.zeros(patch.shape[:2])
            img1[mask1==1] = 1 
            img2[mask2==1] = 1 
            
            pos_contour = measure.find_contours(img1, 0.5)
            neg_contour = measure.find_contours(img2, 0.5)
            for c in pos_contour:
                ax.plot(c[:,1], c[:,0], linewidth=4, color='red')
            for c in neg_contour:
                ax.plot(c[:,1], c[:,0], linewidth=4,color='green')
            ax.imshow(ori_img[..., [0,1,2]])


            plt.axis('off')
            plt.gcf().set_size_inches(512 / 100, 512 / 100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
            plt.margins(0, 0)

            save_path = "YOUR_SAVE_PATH"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path,bbox_inches="tight", pad_inches=0.0)
            plt.close()
            
