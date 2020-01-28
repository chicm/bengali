import os
import pandas as pd
import numpy as np
import time, gc
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pretrainedmodels
from argparse import Namespace
from sklearn.utils import shuffle
from apex import amp
from sklearn.model_selection import StratifiedKFold
from efficientnet_pytorch import EfficientNet
import net

DATA_DIR = '/mnt/chicm/data/bengali'
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test.csv')
class_map_df = pd.read_csv(f'{DATA_DIR}/class_map.csv')
sample_sub_df = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

#plt.imshow(x)
#plt.imshow(x)
import albumentations as albu

def get_train_augs(p=1.):
    return albu.Compose([
        #albu.HorizontalFlip(.5),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5 ),
        albu.Blur(blur_limit=3, p=0.3),
        albu.OpticalDistortion(p=0.3),
        albu.GaussNoise(p=0.3)
        #albu.GridDistortion(p=.33),
        #albu.HueSaturationValue(p=.33) # not for grey scale
    ], p=p)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HEIGHT = 137
WIDTH = 236
SIZE = 224

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return img
    #return cv2.resize(img,(size,size))

class BengaliDataset(Dataset):
    def __init__(self, df, img_df, train_mode=True, test_mode=False):
        self.df = df
        self.img_df = img_df
        self.train_mode = train_mode
        self.test_mode = test_mode

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.get_img(row.image_id)
        #img = crop_resize(img, pad=0)
    
        #print(img.shape)
        if self.train_mode:
            augs = get_train_augs()
            img = augs(image=img)['image']

        #img  = cv2.resize(img, (SIZE, SIZE))
        
        img = np.expand_dims(img, axis=-1)

        img = np.concatenate([img, img, img], 2)
        #print('>>>', img.shape)
        
        # taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128
        MEAN = [ 0.06922848809290576,  0.06922848809290576,  0.06922848809290576]
        STD = [ 0.20515700083327537,  0.20515700083327537,  0.20515700083327537]

        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=MEAN, std=STD)
        
        if self.test_mode:
            return img
        else:
            return img, torch.tensor([row.grapheme_root, row.vowel_diacritic, row.consonant_diacritic])

    def get_img(self, img_id):
        return 255 - self.img_df.loc[img_id].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.df)
    
def get_train_val_loaders(batch_size=4, val_batch_size=4, ifold=0, dev_mode=False):
    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    train_df = shuffle(train_df, random_state=1234)
    print(train_df.shape)

    if dev_mode:
        img_df = pd.read_parquet(f'{DATA_DIR}/train_image_data_0.parquet').set_index('image_id')
        train_df = train_df.iloc[:1000]
    else:
        img_dfs = [pd.read_parquet(f'{DATA_DIR}/train_image_data_{i}.parquet') for i in range(4)]
        img_df = pd.concat(img_dfs, axis=0).set_index('image_id')
    print(img_df.shape)
    #split_index = int(len(train_df) * 0.9)
    
    #train = train_df.iloc[:split_index]
    #val = train_df.iloc[split_index:]
    
    kf = StratifiedKFold(5, random_state=1234, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['grapheme_root'].values)):
        if i == ifold:
            #print(val_idx)
            train = train_df.iloc[train_idx]
            val = train_df.iloc[val_idx]
            break
    assert i == ifold
    print(train.shape, val.shape)
    
    train_ds = BengaliDataset(train, img_df, True, False)
    val_ds = BengaliDataset(val, img_df, False, False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader.num = len(train_ds)

    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=8, drop_last=False)
    val_loader.num = len(val_ds)

    return train_loader, val_loader

class BengaliNet(nn.Module):
    def __init__(self, backbone_name):
        super(BengaliNet, self).__init__()
        self.n_grapheme = 168
        self.n_vowel = 11
        self.n_consonant = 7
        self.backbone_name = backbone_name
        
        self.num_classes = self.n_grapheme + self.n_vowel + self.n_consonant
        
        #self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        
        if self.backbone_name.startswith('efficient'):
            self.backbone = EfficientNet.from_pretrained(self.backbone_name)
            self.fc = nn.Linear(self.backbone._fc.in_features, self.num_classes)
        elif self.backbone_name in net.__dict__:
            self.backbone = net.__dict__[self.backbone_name](pretrained=True)

            if self.backbone_name.startswith('densenet') or self.backbone_name.startswith('squeeze'):
                in_features = self.backbone.classifier.in_features
            elif self.backbone_name.startswith('mobilenet'):
                in_features = self.backbone.last_channel
            else:
                in_features = self.backbone.fc.in_features
            self.fc = nn.Linear(in_features, self.num_classes)
        else:
            self.backbone = pretrainedmodels.__dict__[self.backbone_name](num_classes=1000, pretrained='imagenet')
            
            if isinstance(self.backbone.last_linear, nn.Conv2d):
                in_features = self.backbone.last_linear.in_channels
            else:
                in_features = self.backbone.last_linear.in_features
            self.fc = nn.Linear(in_features, self.num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #self.fix_input_layer()s
        
    def fix_input_layer(self):
        #if self.backbone_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'senet154', 'se_resnet152', 'nasnetmobile', 'mobilenet', 'nasnetalarge']:
        if not self.backbone_name.startswith('efficient'):
            #self.backbone = eval(backbone_name)()
            #print(self.backbone.layer0.conv1)
            w = self.backbone.layer0.conv1.weight.data
            self.backbone.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #self.backbone.layer0.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, 2, :, :].unsqueeze(1)), dim=1))
            self.backbone.layer0.conv1.weight = torch.nn.Parameter(w[:, 0, :, :].unsqueeze(1))
        
    def logits(self, x):
        x = self.avg_pool(x)
        #x = F.dropout2d(x, 0.2, self.training)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def forward(self, x):
        #x = self.conv0(x)
        #print(x.size())
        if self.backbone_name.startswith('efficient'):
            x = self.backbone.extract_features(x)
        else:
            x = self.backbone.features(x)
        x = self.logits(x)

        return x

MODEL_DIR = './models'
def create_model(args):
    model = BengaliNet(backbone_name=args.backbone)
    model_file = os.path.join(MODEL_DIR, args.backbone, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    
    return model, model_file


import numpy as np
import sklearn.metrics
import torch


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    #y = y.cpu().numpy()
    # pred_y = [p.cpu().numpy() for p in pred_y]

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y_grapheme, average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y_vowel, average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y_consonant, average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}, y {y.shape}')
    return final_score

def calc_metrics(preds0, preds1, preds2, y):
    assert len(y) == len(preds0) == len(preds1) == len(preds2)

    recall_grapheme = sklearn.metrics.recall_score(preds0, y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(preds1, y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(preds2, y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_recall_score = np.average(scores, weights=[2, 1, 1])
    
    metrics = {}
    metrics['recall'] = round(final_recall_score, 6)
    metrics['recall_grapheme'] = round(recall_grapheme, 6)
    metrics['recall_vowel'] = round(recall_vowel, 6)
    metrics['recall_consonant'] = round(recall_consonant, 6)
    
    metrics['acc_grapheme'] = round((preds0 == y[:, 0]).sum() / len(y), 6)
    metrics['acc_vowel'] = round((preds1 == y[:, 1]).sum() / len(y), 6)
    metrics['acc_consonant'] = round((preds2 == y[:, 2]).sum() / len(y), 6)
    
    
    return metrics

def criterion(outputs, y_true):
    # outputs: (N, 182)
    # y_true: (N, 3)
    
    outputs = torch.split(outputs, [168, 11, 7], dim=1)
    loss0 = F.cross_entropy(outputs[0], y_true[:, 0], reduction='mean')
    loss1 = F.cross_entropy(outputs[1], y_true[:, 1], reduction='mean')
    loss2 = F.cross_entropy(outputs[2], y_true[:, 2], reduction='mean')
    
    return loss0 + loss1 + loss2 #, loss0.item(), loss1.item(), loss2.item()

def validate(model, val_loader):
    model.eval()
    loss0, loss1, loss2 = 0., 0., 0.
    preds0, preds1,preds2 = [], [], []
    y_true = []
    with torch.no_grad():
        for x, y in val_loader:
            y_true.append(y)
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            outputs = torch.split(outputs, [168, 11, 7], dim=1)
            
            preds0.append(torch.max(outputs[0], dim=1)[1])
            preds1.append(torch.max(outputs[1], dim=1)[1])
            preds2.append(torch.max(outputs[2], dim=1)[1])
            loss0 += F.cross_entropy(outputs[0], y[:, 0], reduction='sum').item()
            loss1 += F.cross_entropy(outputs[1], y[:, 1], reduction='sum').item()
            loss2 += F.cross_entropy(outputs[2], y[:, 2], reduction='sum').item()
            
            # for debug
            #metrics = {}
            #metrics['loss_grapheme'] =  F.cross_entropy(outputs[0], y[:, 0], reduction='mean').item()
            #metrics['loss_vowel'] =  F.cross_entropy(outputs[1], y[:, 1], reduction='mean').item()
            #metrics['loss_consonant'] =  F.cross_entropy(outputs[2], y[:, 2], reduction='mean').item()
            #return metrics
    
    preds0 = torch.cat(preds0, 0).cpu().numpy()
    preds1 = torch.cat(preds1, 0).cpu().numpy()
    preds2 = torch.cat(preds2, 0).cpu().numpy()
    y_true = torch.cat(y_true, 0).numpy()
    
    #print('y_true:', y_true.shape)
    #print('preds0:', preds0.shape)
    
    metrics = calc_metrics(preds0, preds1, preds2, y_true)
    metrics['loss_grapheme'] = round(loss0 / val_loader.num, 6)
    metrics['loss_vowel'] = round(loss1 / val_loader.num, 6)
    metrics['loss_consonant'] = round(loss2 / val_loader.num, 6)
    
    return metrics

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs
def save_model(model, model_file):
    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_file)
    else:
        torch.save(model.state_dict(), model_file)

def mixup(data, targets, alpha=1):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = (targets, shuffled_targets, lam)

    return data, targets


def mixup_criterion(outputs, targets):
    targets1, targets2, lam = targets
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

from over9000.over9000 import Over9000
from over9000.radam import RAdam

def train(model, train_loader, val_loader, args):
    #global model

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optim == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr)
    elif args.optim == 'Over9000':
        optimizer = Over9000(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_metrics = 0.
    best_key = 'recall'
    
    val_metrics = validate(model, val_loader)
    print(val_metrics)
    best_metrics = val_metrics[best_key]
    
    model.train()
    #optimizer.zero_grad()

    #if args.lrs == 'plateau':
    #    lr_scheduler.step(best_metrics)
    #else:
    #    lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.num_epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)
        bg = time.time()
        for batch_idx, (img, targets) in enumerate(train_loader):
            train_iter += 1
            img, targets  = img.cuda(), targets.cuda()
            #do_mixup = False #(np.random.random() < 0.4)
            
            #if do_mixup:
            #    img, targets = mixup(img, targets)
            batch_size = img.size(0)
          
            
            
            #if do_mixup:
            #    loss = mixup_criterion(outputs, targets)
            #else:
            #    loss = criterion(outputs, targets)
            r = np.random.rand()
            #if args.beta > 0 and r < args.cutmix_prob:
            if r < 0.5:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(img.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                # compute output
                outputs = model(img)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                #img, targets = mixup(img, targets)
                outputs = model(img)
                #loss = mixup_criterion(outputs, targets)
                loss = criterion(outputs, targets)
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            #loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            #if batch_idx % 4 == 0:
            #    optimizer.step()
            #    optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), batch_size*(batch_idx+1), train_loader.num, 
                loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                #outputs = torch.split(outputs, [168, 11, 7], dim=1)
            
                #preds0 = (torch.max(outputs[0], dim=1)[1]).cpu().numpy()
                #preds1 = (torch.max(outputs[1], dim=1)[1]).cpu().numpy()
                #preds2 = (torch.max(outputs[2], dim=1)[1]).cpu().numpy()
                #train_metrics = calc_metrics(preds0, preds1, preds2, targets.cpu().numpy())
                #print('train:', train_metrics)
                #save_model(model, model_file+'_latest')
                val_metrics = validate(model, val_loader)
                print('\nval:', val_metrics)

                nni.report_intermediate_result(val_metrics['recall'])
                
                if val_metrics[best_key] > best_metrics:
                    best_metrics = val_metrics[best_key]
                    save_model(model, model_file)
                    print('** saved')
                
                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(best_metrics)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    nni.report_final_result(best_metrics)

import nni

# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2', 
#    'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 
# 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
#  'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetamobile', 'nasnetalarge', 'dpn68', 'dpn68b',
# 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'xception', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
# 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'pnasnet5large', 'polynet']

hyper_params = nni.get_next_parameter()

args = Namespace()
args.backbone = hyper_params['backbone']
args.ckp_name = 'model2_fold0.pth'
args.predict = False
args.optim = 'RAdam'
args.lr = 1e-3
args.lrs = 'plateau'
args.t_max = 15
args.factor = 0.1
args.patience = 0
args.min_lr = 1e-5
args.iter_val = 100
args.num_epochs = 5
args.batch_size = 512
args.val_batch_size = 1024

args.beta = 1.0
args.cutmix_prob = 0.5

train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, ifold=0)

#for img, label in train_loader:
#    print(img.size())
#    print(label.size())
#    break
model, model_file = create_model(args)
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
model = model.cuda()

train(model, train_loader, val_loader, args)
