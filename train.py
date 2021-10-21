import numpy as np
from glob import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from skimage.io import imread
from PIL import Image
import copy
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PDL1 segmentation')
parser.add_argument('--train_img', type=str, default='', help='path to training set of original images.')
parser.add_argument('--val_img', type=str, default='', help='path to validation set of original images.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--size', type=int, default=416, help='input size of training img')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')


def tversky_loss(true, inputs, alpha=0.7, eps=1e-7):
    y_true_pos = inputs.contiguous().view(-1)
    y_pred_pos = true.contiguous().view(-1)
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
    false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
    score = (true_pos + eps) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + eps)
    return (1 - score)


def tverskyClassWeighted(pred, target, class_weights=None):
    added_weights = class_weights.sum()

    batch_loss = torch.tensor(0.0).cuda()
    for instance_output, instance_target in zip(pred, target):
        instance_loss = torch.tensor(0.0).cuda()
        for out_channel_output, out_channel_target, weight in zip(instance_output, instance_target,
                                                                  class_weights):
            instance_loss += weight * tversky_loss(out_channel_output, out_channel_target)
        batch_loss += instance_loss / added_weights
    return batch_loss / len(pred)


def dice_loss(pred, target, smooth=1e-2):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    # print(iflat.shape)
    intersection = (iflat * tflat).sum()
    # print('intersection=',intersection)
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    # A_sum = (iflat * iflat).sum()
    # B_sum = (tflat * tflat).sum()
    # print('B_sum=',B_sum)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def diceClassWeighted(pred, target, class_weights=None):
    added_weights = class_weights.sum()

    batch_loss = torch.tensor(0.0).cuda()
    # split batch to instance
    for instance_output, instance_target in zip(pred, target):
        instance_loss = torch.tensor(0.0).cuda()
        # split channel to each class
        for out_channel_output, out_channel_target, weight in zip(instance_output,
                                                                  instance_target,
                                                                  class_weights):
            instance_loss += weight * dice_loss(out_channel_output, out_channel_target)
        batch_loss += instance_loss / added_weights
    return batch_loss / pred.size(0)


class my_dataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None, target_transform=None):
        data_path = []
        for img_path in img_list:
            # mask_path = img_path.replace('train', 'mask')
            # mask_path = mask_path.replace('val', 'mask')

            mask_path = img_path.replace('_img', '_mask')
            mask_path = mask_path.replace('jpg', 'npy')
            data_path.append([img_path, mask_path])

        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.data_path[index]
        img_x = imread(x_path)
        img_y = np.load(y_path)
        img_y = np.array(img_y, np.uint8)
        img_x = Image.fromarray(img_x)
        img_y = Image.fromarray(img_y)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.data_path)


##################################### The construction of Unet ##############################################
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


class UNet(nn.Module):
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
        # out = torch.sigmoid(out)
        return out


def train_model(model, optimizer, scheduler, criterion, image_datasets, num_epochs=25):
    since = time.time()
    model = model.cuda()
    best_loss = 100.0
    best_epoch = 0

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.workers)
                   for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        miou_and_1 = 0.0
        miou_and_2 = 0.0
        miou_or_1 = 0.0
        miou_or_2 = 0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if phase == 'val':
                    mask_preds = outputs
                    gt_preds = labels

                    gt_preds[gt_preds > 0.5] = 1.
                    gt_preds[gt_preds < 0.5] = 0.
                    mask_preds[mask_preds > 0.5] = 1.
                    mask_preds[mask_preds < 0.5] = 0.
                    area1 = gt_preds[:, 0, ...] + mask_preds[:, 0, ...]
                    area2 = gt_preds[:, 1, ...] + mask_preds[:, 1, ...]

                    miou_and_1 += torch.sum(area1 == 2).item()
                    miou_and_2 += torch.sum(area2 == 2).item()
                    miou_or_1 += (
                            torch.sum(gt_preds[:, 0, ...] == 1) + torch.sum(mask_preds[:, 0, ...] == 1) - torch.sum(
                        area1 == 2)).item()
                    miou_or_2 += (
                            torch.sum(gt_preds[:, 1, ...] == 1) + torch.sum(mask_preds[:, 1, ...] == 1) - torch.sum(
                        area2 == 2)).item()

                # class_weight = torch.tensor([1, 3]).cuda()
                # loss = criterion(outputs, labels, class_weight)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,
                           args.output + '/best_loss{}_epoch{}.pth'.format(round(best_loss, 4), best_epoch))

            if phase == 'val':
                if miou_or_1 > 0:
                    print('avg_miou1: {:.4f}'.format(miou_and_1 / miou_or_1))
                else:
                    print('avg_miou1: /')
                if miou_or_2 > 0:
                    print('avg_miou2: {:.4f}'.format(miou_and_2 / miou_or_2))
                else:
                    print('avg_miou2: /')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val epoch: {:.4f}'.format(best_epoch))
    return model


def main():
    global args
    args = parser.parse_args()
    train_img_path = args.train_img
    val_img_path = args.val_img
    epochs = args.nepochs
    input_size = args.size
    n_classes = args.nclasses

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            # transforms.ColorJitter([0.8, 1.3], 0.2, 0.2, 0.05),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),

        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
    }

    label_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    train_img = glob(train_img_path + '/*.jpg')
    val_img = glob(val_img_path + '/*.jpg')

    train_dataset = my_dataset(train_img, transform=data_transforms["train"],
                               target_transform=label_transforms)
    val_dataset = my_dataset(val_img, transform=data_transforms["val"],
                             target_transform=label_transforms)

    image_datasets = {"train": train_dataset, "val": val_dataset}

    criterion = dice_loss
    # criterion = diceClassWeighted
    model_ft = UNet(n_classes)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    model_train = train_model(model_ft, optimizer_ft, exp_lr_scheduler, criterion, image_datasets,
                              num_epochs=epochs)


if __name__ == '__main__':
    main()
