import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import timm
#from timm.models.layers.activations import Swish, Mish
from timm.models.activations import Swish, Mish

from archead import *


class Enet(nn.Module):
    def __init__(self, num_layers=0, num_classes1=7, num_classes2=168, num_classes3=11, pretrained=True, dropout=False):
        super().__init__()
        # self.base = EfficientNet.from_name('efficientnet-b{}'.format(num_layers))
        self.base = EfficientNet.from_pretrained('efficientnet-b{}'.format(num_layers), in_channels=1)
        self.model_name = 'efficientnet-b{}'.format(num_layers)
        # if pretrained:
        #     self.base.load_state_dict(torch.load('/data2/liufuxu/projects/aptos/pretrained/b{}.pth'.format(num_layers)))
        # self.base._conv_stem = Conv2dStaticSamePadding(1, 48, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=456)
        self.in_features = self.base._fc.in_features

        # self.avg = nn.AdaptiveMaxPool2d(1)
        if dropout:
            self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.in_features, 168))    ## consonant_diacritic->7
            self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.in_features, 11))     ## grapheme_root->168  
            self.fc3 = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.in_features, 7))      ## vowel_diacritic ->11
        else:
            self.fc1 = nn.Linear(self.in_features, 168)    ## consonant_diacritic->7
            self.fc2 = nn.Linear(self.in_features, 11)    ## grapheme_root->168  
            self.fc3 = nn.Linear(self.in_features, 7)    ## vowel_diacritic ->11

    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        # x = F.dropout(x, 0.5, self.training)
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        return logit1, logit2, logit3


class Enet_timm(nn.Module):
    def __init__(self, num_layers=5, pretrained=True):
        super().__init__()
        self.base = timm.create_model(model_name='tf_efficientnet_b{}'.format(num_layers), pretrained=pretrained, in_chans=1, drop_connect_rate=0.2)
        self.model_name = 'efficientnet-b{}'.format(num_layers)
        
        self.in_features = self.base.num_features
        
        self.conv_stem = self.base.conv_stem
        self.bn1 = self.base.bn1
        self.act1 = self.base.act1
        self.blocks = self.base.blocks
        self.conv_head = self.base.conv_head
        self.bn2 = self.base.bn2
        self.act2 = self.base.act2
        self.global_pool = self.base.global_pool
        
        self.fc1 = nn.Linear(self.in_features, 168)    ## consonant_diacritic->7
        self.fc2 = nn.Linear(self.in_features, 11)    ## grapheme_root->168  
        self.fc3 = nn.Linear(self.in_features, 7)    ## vowel_diacritic ->11

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        return logit1, logit2, logit3


class Enet_timm_aux(nn.Module):
    def __init__(self, num_layers=5, pretrained=True):
        super().__init__()
        self.base = timm.create_model(model_name='tf_efficientnet_b{}'.format(num_layers), pretrained=pretrained, in_chans=1, drop_connect_rate=0.2)
        self.model_name = 'efficientnet-b{}'.format(num_layers)
        
        self.in_features = self.base.num_features
        
        self.conv_stem = self.base.conv_stem
        self.bn1 = self.base.bn1
        self.act1 = self.base.act1
        self.blocks0 = self.base.blocks[0]
        self.blocks1 = self.base.blocks[1]
        self.blocks2 = self.base.blocks[2]
        self.blocks3 = self.base.blocks[3]
        self.blocks4 = self.base.blocks[4]
        self.blocks5 = self.base.blocks[5]
        self.blocks6 = self.base.blocks[6]
        self.conv_head = self.base.conv_head
        self.bn2 = self.base.bn2
        self.act2 = self.base.act2
        self.global_pool = self.base.global_pool
        
        self.fc1 = nn.Linear(self.in_features, 168)    ## consonant_diacritic->7
        self.fc2 = nn.Linear(self.in_features, 11)    ## grapheme_root->168  
        self.fc3 = nn.Linear(self.in_features, 7)    ## vowel_diacritic ->11

        self.aux1_fc1 = nn.Linear(176, 168)    ## consonant_diacritic->7
        self.aux1_fc2 = nn.Linear(176, 11)    ## grapheme_root->168  
        self.aux1_fc3 = nn.Linear(176, 7)    ## vowel_diacritic ->11

        self.aux2_fc1 = nn.Linear(304, 168)    ## consonant_diacritic->7
        self.aux2_fc2 = nn.Linear(304, 11)    ## grapheme_root->168  
        self.aux2_fc3 = nn.Linear(304, 7)    ## vowel_diacritic ->11

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks0(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x); b4 = x
        x = self.blocks5(x); b5 = x
        x = self.blocks6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.global_pool(x)

        if self.training:
            b4 = self.global_pool(b4)
            b5 = self.global_pool(b5)
            b4 = b4.view(b4.size(0), -1)
            b5 = b5.view(b5.size(0), -1)

            aux1_logit1 = self.aux1_fc1(b4)
            aux1_logit2 = self.aux1_fc2(b4)
            aux1_logit3 = self.aux1_fc3(b4)

            aux2_logit1 = self.aux2_fc1(b5)
            aux2_logit2 = self.aux2_fc2(b5)
            aux2_logit3 = self.aux2_fc3(b5)

        x = x.view(x.size(0), -1)
        
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        if self.training:
            return [logit1, logit2, logit3], [aux1_logit1, aux1_logit2, aux1_logit3], [aux2_logit1, aux2_logit2, aux2_logit3],
        else:
            return logit1, logit2, logit3

class Enet_timm_aux_arc(nn.Module):
    def __init__(self, num_layers=5, pretrained=True):
        super().__init__()
        self.base = timm.create_model(model_name='tf_efficientnet_b{}'.format(num_layers), pretrained=pretrained, in_chans=1, drop_connect_rate=0.2)
        self.model_name = 'efficientnet-b{}'.format(num_layers)
        
        self.in_features = self.base.num_features
        
        self.conv_stem = self.base.conv_stem
        self.bn1 = self.base.bn1
        self.act1 = self.base.act1
        self.block0 = self.base.blocks[0]
        self.block1 = self.base.blocks[1]
        self.block2 = self.base.blocks[2]
        self.block3 = self.base.blocks[3]
        self.block4 = self.base.blocks[4]
        self.block5 = self.base.blocks[5]
        self.block6 = self.base.blocks[6]
        
        self.conv_head = self.base.conv_head
        self.bn2 = self.base.bn2
        self.act2 = self.base.act2
        self.global_pool = self.base.global_pool
        
        self.aux_block5 = self.block5
        self.aux_num_features = self.block5[-1].bn3.num_features
        self.aux_head4 = nn.Conv2d(self.aux_num_features, self.aux_num_features * 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(self.aux_num_features * 4)
        self.act4 = Swish()
        self.aux_head5 = nn.Conv2d(self.aux_num_features, self.aux_num_features * 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(self.aux_num_features * 4)
        self.act5 = Swish()
        
        ## norm linear
        self.fc1 = nn.Linear(self.in_features, 168)    ## consonant_diacritic->7
        self.fc2 = nn.Linear(self.in_features, 11)    ## grapheme_root->168  
        self.fc3 = nn.Linear(self.in_features, 7)    ## vowel_diacritic ->11
        self.fc4 = nn.Linear(self.in_features, 1295)
        
        self.aux1_fc1 = nn.Linear(self.aux_num_features * 4, 168)    ## consonant_diacritic->7
        self.aux1_fc2 = nn.Linear(self.aux_num_features * 4, 11)    ## grapheme_root->168  
        self.aux1_fc3 = nn.Linear(self.aux_num_features * 4, 7)    ## vowel_diacritic ->11
        self.aux1_fc4 = nn.Linear(self.aux_num_features * 4, 1295)
        
        
        self.aux2_fc1 = nn.Linear(self.aux_num_features * 4, 168)    ## consonant_diacritic->7
        self.aux2_fc2 = nn.Linear(self.aux_num_features * 4, 11)    ## grapheme_root->168  
        self.aux2_fc3 = nn.Linear(self.aux_num_features * 4, 7)    ## vowel_diacritic ->11
        self.aux2_fc4 = nn.Linear(self.aux_num_features * 4, 1295)
        
        
        ## arcface
        self.arc_rt1 = ArcMarginProduct(self.in_features, 168, s=30, m=0.2)
        self.arc_vow1= ArcMarginProduct(self.in_features, 11, s=30, m=0.2)
        self.arc_con1 = ArcMarginProduct(self.in_features, 7, s=30, m=0.2)
        self.arc_word1 = ArcMarginProduct(self.in_features, 1295, s=30, m=0.2)
        
        self.arc_rt2 = ArcMarginProduct(self.aux_num_features * 4, 168, s=30, m=0.2)
        self.arc_vow2 = ArcMarginProduct(self.aux_num_features * 4, 11, s=30, m=0.2)
        self.arc_con2 = ArcMarginProduct(self.aux_num_features * 4, 7, s=30, m=0.2)
        self.arc_word2 = ArcMarginProduct(self.aux_num_features * 4, 1295, s=30, m=0.2)
                                           
        self.arc_rt3 = ArcMarginProduct(self.aux_num_features * 4, 168, s=30, m=0.2)
        self.arc_vow3 = ArcMarginProduct(self.aux_num_features * 4, 11, s=30, m=0.2)
        self.arc_con3 = ArcMarginProduct(self.aux_num_features * 4, 7, s=30, m=0.2)
        self.arc_word3 = ArcMarginProduct(self.aux_num_features * 4, 1295, s=30, m=0.2)
                                                                               
    def forward(self, x, label = None):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b4 = self.aux_block5(b4); b5 = x
        x = self.block6(x)
        x = self.conv_head(x); b4 = self.aux_head4(b4); b5 = self.aux_head5(b5)
        x = self.bn2(x); b4 = self.bn4(b4); b5 = self.bn5(b5)
        x = self.act2(x); b4 = self.act4(b4); b5 = self.act5(b5)

        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        
        if self.training:
            logit4 = self.fc4(x)
        
            b4 = self.global_pool(b4)
            b5 = self.global_pool(b5)
            b4 = b4.view(b4.size(0), -1)
            b5 = b5.view(b5.size(0), -1)

            aux1_logit1 = self.aux1_fc1(b4)
            aux1_logit2 = self.aux1_fc2(b4)
            aux1_logit3 = self.aux1_fc3(b4)
            aux1_logit4 = self.aux1_fc4(b4)
                                         
            aux2_logit1 = self.aux2_fc1(b5)
            aux2_logit2 = self.aux2_fc2(b5)
            aux2_logit3 = self.aux2_fc3(b5)
            aux2_logit4 = self.aux2_fc4(b5)
            
            ## arcface
            arc_logit1 = self.arc_rt1(x, label[0])
            arc_logit2 = self.arc_vow1(x, label[1])
            arc_logit3 = self.arc_con1(x, label[2])
            arc_logit4 = self.arc_word1(x, label[3])

            arc_aux1_logit1 = self.arc_rt2(b4, label[0])
            arc_aux1_logit2 = self.arc_vow2(b4, label[1])
            arc_aux1_logit3 = self.arc_con2(b4, label[2])
            arc_aux1_logit4 = self.arc_word2(b4, label[3])

            arc_aux2_logit1 = self.arc_rt3(b5, label[0])
            arc_aux2_logit2 = self.arc_vow3(b5, label[1])
            arc_aux2_logit3 = self.arc_con3(b5, label[2])
            arc_aux2_logit4 = self.arc_word3(b5, label[3])
            
        if self.training:
            return [logit1, logit2, logit3, logit4], [aux1_logit1, aux1_logit2, aux1_logit3, aux1_logit4], [aux2_logit1, aux2_logit2, aux2_logit3, aux2_logit4], \
        [arc_logit1, arc_logit2, arc_logit3, arc_logit4], [arc_aux1_logit1, arc_aux1_logit2, arc_aux1_logit3, arc_aux1_logit4], \
        [arc_aux2_logit1, arc_aux2_logit2, arc_aux2_logit3, arc_aux2_logit4]
        else:
            return logit1, logit2, logit3

    def set_mode(self, mode, is_freeze_bn=False):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()

        elif mode in ['train']:
            self.train()
        if is_freeze_bn == True:  ##freeze
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False