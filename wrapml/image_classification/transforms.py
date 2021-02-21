import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size, additional_transforms=[]):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(.3, 1), p=1),
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        *additional_transforms,
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255, always_apply=True
        ),
        ToTensorV2()
    ])

def get_valid_transforms(img_size):
    return A.Compose([
        A.PadIfNeeded(img_size, img_size),
        A.CenterCrop(img_size, img_size, p=0.5),
        A.Resize(img_size, img_size, p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255, always_apply=True
        ),
        ToTensorV2()
    ])

def get_test_transforms(img_size):
    return A.Compose([
        A.PadIfNeeded(img_size, img_size),
        A.CenterCrop(img_size, img_size, p=0.5),
        A.Resize(img_size, img_size, p=1),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255, always_apply=True
        ),
        ToTensorV2()
    ])



# SnapMix from https://github.com/Shaoli-Huang/SnapMix

def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_spm(X, y, feature_extracter, classifier, img_size):
    imgsize = (img_size, img_size)
    bs = X.size(0)
    with torch.no_grad():
        fms = feature_extracter(X)
        pooled_features = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        # currently only timm efficientnet is supported
        # changes might be needed for other models
        # if 'inception' in conf.netname:
        #     clsw = model.module.fc
        # else:

        # it is assumed that the last layer of classifier is fc
        clsw = classifier[-1]
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        clslogit = F.softmax(clsw.forward(pooled_features))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i,y[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,y[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps, clslogit


def snapmix(X, y, feature_extracter, classifier, img_size, beta=1):
    lam_a = torch.ones(X.size(0))
    lam_b = 1 - lam_a
    y_b = y.clone()
    wfmaps,_ = get_spm(X, y, feature_extracter, classifier, img_size)
    bs = X.size(0)
    lam = np.random.beta(beta, beta)
    lam1 = np.random.beta(beta, beta)
    rand_index = torch.randperm(bs).cuda()
    wfmaps_b = wfmaps[rand_index,:,:]
    y_b = y[rand_index]

    same_label = y == y_b
    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(X.size(), lam1)

    area = (bby2-bby1)*(bbx2-bbx1)
    area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

    if  area1 > 0 and  area>0:
        ncont = X[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
        X[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
        lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
        lam_a[torch.isnan(lam_a)] = lam
        lam_b[torch.isnan(lam_b)] = 1-lam

    return X,y,y_b,lam_a.cuda(),lam_b.cuda()