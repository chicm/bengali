import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from dataset_cutmix import *
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import recall_score
from timeit import default_timer as timer
from efficientnet import *
from gridmask import GridMask
from apex import amp
import apex

grid = GridMask(64, 128, rotate=15, ratio=0.6, mode=1, prob=1.)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

def do_valid(net, valid_loader, loss_function):
    valid_num  = 0
    losses   = []
    probs = [[],[], []]
    labels = [[],[], []]
    net.eval()
    
    with torch.no_grad():
        for input, lb1, lb2, lb3, lb4 in valid_loader:
            
            input, lb1, lb2, lb3 = input.cuda(), lb1.cuda(), lb2.cuda(), lb3.cuda()
            out1, out2, out3 = net.forward(input)
            #calculate
            loss1, loss2, loss3 = \
            loss_function(out1, lb1), loss_function(out2, lb2), loss_function(out3, lb3)
            #prepare for output
            loss_all = 2*loss1 + loss2 + loss3
            loss_all = loss_all.data.cpu().numpy().reshape([1])

            losses.append(loss_all)

            res1 = F.softmax(out1, dim=1).data.cpu().numpy().argmax(axis=1)
            res2 = F.softmax(out2, dim=1).data.cpu().numpy().argmax(axis=1)
            res3 = F.softmax(out3, dim=1).data.cpu().numpy().argmax(axis=1)

            probs[0].append(res1)
            probs[1].append(res2)
            probs[2].append(res3)

            labels[0].append(lb1.data.cpu().numpy())
            labels[1].append(lb2.data.cpu().numpy())
            labels[2].append(lb3.data.cpu().numpy())

            valid_num += len(input)

    assert (valid_num == len(valid_loader.sampler))
    # ------------------------------------------------------
    loss = np.concatenate(losses,axis=0)
    loss = loss.mean()

    probs[0] = np.concatenate(probs[0],axis=0)
    probs[1] = np.concatenate(probs[1],axis=0)
    probs[2] = np.concatenate(probs[2],axis=0)

    labels[0] = np.concatenate(labels[0],axis=0)
    labels[1] = np.concatenate(labels[1],axis=0)
    labels[2] = np.concatenate(labels[2],axis=0)

    score1 = recall_score(labels[0], probs[0], average='macro')
    score2 = recall_score(labels[1], probs[1], average='macro')
    score3 = recall_score(labels[2], probs[2], average='macro')

    score_all = np.average([score1, score2, score3], weights=[2,1,1])
    return loss, score_all, score1, score2, score3

###mix up related###
def to_onehot(truth, num_class):
    batch_size = len(truth)
    onehot = torch.zeros(batch_size,num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1,1),value=1)
    return onehot

def cross_entropy_onehot_loss(logit, onehot):
    batch_size,num_class = logit.shape
    log_probability = -F.log_softmax(logit,1)
    loss = (log_probability*onehot)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss

def criterion(logit, truth, lam):
    loss = []
    for l,t in zip(logit,truth):
        #e = F.cross_entropy(l, t)
        e = cross_entropy_onehot_loss(l, t)*lam
        loss.append(e)
    return loss

def do_mixup(input, onehot):
    batch_size = len(input)

    alpha = 0.4  #0.2,0.4
    gamma = np.random.beta(alpha, alpha)
    gamma = max(1-gamma,gamma)

    # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
    perm = torch.randperm(batch_size).to(input.device)
    perm_input  = input[perm]
    perm_onehot = [t[perm] for t in onehot]
    mix_input  = gamma*input + (1-gamma)*perm_input
    mix_onehot = [gamma*t    + (1-gamma)*perm_t for t,perm_t in zip(onehot,perm_onehot)]
    return mix_input, mix_onehot
###mix up related###
def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
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

def run_train(config):
    base_lr = config.base_lr
    
    def adjust_lr_and_hard_ratio(optimizer, ep):
        if ep < 50:
            lr = 4e-4
        elif ep < 100:
            lr = 1e-4
        else:
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr
        return lr

    #basic info.
    batch_size = config.batch_size
    image_size = (config.image_h, config.image_w)
    meta_df = pd.read_csv(config.meta_df)

    grapheme_words = np.unique(meta_df.grapheme.values)
    grapheme_words_dict = {grapheme: i for i, grapheme in enumerate(grapheme_words)}
    meta_df['word_label'] = meta_df['grapheme'].map(lambda x: grapheme_words_dict[x])

    #img_path = config.img_path
    fold = config.fold
    ## setup  -----------------------------------------------------------------------------
    out_dir = os.path.join('./ckpt/', config.model)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir,'checkpoint')):
        os.makedirs(os.path.join(out_dir,'checkpoint'))
    if not os.path.exists(os.path.join(out_dir,'train')):
        os.makedirs(os.path.join(out_dir,'train'))

    if config.pretrained_model is not None:
        initial_checkpoint = os.path.join(out_dir, config.pretrained_model)
    else:
        initial_checkpoint = None

    print('loading parquets...')
    img_dfs = [pd.read_parquet(f'{config.data_dir}/train_image_data_{i}.parquet') for i in range(4)]
    img_df = pd.concat(img_dfs, axis=0).set_index('image_id')
    print('done,',  img_df.shape)

    train_dataset = GraphemeDataset_aux(meta_df, 'train', img_df, image_size=image_size, fold=fold)

    train_loader  = DataLoader(train_dataset,
                                shuffle = True,
                                batch_size  = batch_size,
                                drop_last   = True,
                                num_workers = config.num_workers,
                                pin_memory  = True)

    valid_dataset = GraphemeDataset_aux(meta_df, 'val', img_df, image_size=image_size, fold=fold)

    valid_loader  = DataLoader(valid_dataset,
                                shuffle = False,
                                batch_size  = batch_size,
                                drop_last   = False,
                                num_workers = config.num_workers,
                                pin_memory  = True)

    net = Enet_timm_aux_arc(num_layers=5, pretrained=True)
    net = apex.parallel.convert_syncbn_model(net)
    net = net.cuda()

    if config.schedule == 'cosine':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), \
            lr=config.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, T_mult=1, eta_min=config.final_lr)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), \
            lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4)

    if config.apex_flag:
        print('training with apex')
        net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)


    net = torch.nn.DataParallel(net)

    log = open(out_dir+'/log.train.txt', mode='a')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    log.write('%s\n'%(type(net)))
    log.write('\n')

    loss_function = nn.CrossEntropyLoss()

    iter_smooth = 20
    start_iter = 0

    log.write('\n')
    log.write('** start training here! **\n')
    print('** start training here! **\n')

    i = 0
    start = timer()
    min_loss, max_score, score1, score2, score3 = do_valid(net, valid_loader, loss_function)
    print('start from: loss {} score {}'.format(min_loss, max_score))
    max_score_print = max_score
    # max_score = 0
    # max_score_print = 0
    iter_per_epoch = train_dataset.num_data / batch_size
    print('iter_per_epoch {}'.format(iter_per_epoch))
    net.train()
    for epoch in range(config.train_epoch):
        optimizer.zero_grad()
        train_all_loss = 0
        if config.schedule == 'cosine': 
            rate = optimizer.state_dict()['param_groups'][0]['lr']
        else:
            rate = adjust_lr_and_hard_ratio(optimizer, epoch + 1)
        
        for input, lb1, lb2, lb3, lb4 in train_loader:
            iter = i + start_iter
            # one iteration update  -------------
            net.train()
            input, lb1, lb2, lb3, lb4 = input.cuda(), lb1.cuda(), lb2.cuda(), lb3.cuda(), lb4.cuda()
            rand_value = np.random.rand()
            if rand_value<0.4:
                #cutmix part#
                lam = np.random.beta(config.beta, config.beta)
                rand_index = np.random.permutation(input.size()[0])
                target_a = [to_onehot(t,c) for t,c in zip([lb1, lb2, lb3, lb4],[168, 11, 7, 1295])]
                target_b = [t[rand_index] for t in target_a]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                output1, output2, output3, output4, output5, output6 = net.forward(input, [lb1, lb2, lb3, lb4])
                loss1 = criterion(output1, target_a, lam) + \
                criterion(output1, target_b, (1. - lam))
                loss2 = criterion(output2, target_a, lam) + \
                criterion(output1, target_b, (1. - lam))
                loss3 = criterion(output3, target_a, lam) + \
                criterion(output1, target_b, (1. - lam))
                
                ## arcface
                loss4 = criterion(output4, target_a, lam) + \
                criterion(output4, target_b, (1. - lam))
                loss5 = criterion(output5, target_a, lam) + \
                criterion(output5, target_b, (1. - lam))
                loss6 = criterion(output6, target_a, lam) + \
                criterion(output6, target_b, (1. - lam))

                #cutmix part#
            elif rand_value<0.7:
                #gridmask part#
                onehot = [to_onehot(t,c) for t,c in zip([lb1, lb2, lb3, lb4],[168, 11, 7, 1295])]
                with torch.no_grad():
                    input = grid(input)
                output1, output2, output3, output4, output5, output6 = net.forward(input, [lb1, lb2, lb3, lb4])

                loss1 = criterion(output1, onehot, 1)
                loss2 = criterion(output2, onehot, 1)
                loss3 = criterion(output3, onehot, 1)

                loss4 = criterion(output4, onehot, 1)
                loss5 = criterion(output5, onehot, 1)
                loss6 = criterion(output6, onehot, 1)
                #gridmask part#
            else:
                #mix up part#
                onehot = [to_onehot(t,c) for t,c in zip([lb1, lb2, lb3],[168, 11, 7, 1295])]
                with torch.no_grad():
                    input, onehot = do_mixup(input, onehot)
                output1, output2, output3, output4, output5, output6 = net.forward(input, [lb1, lb2, lb3, lb4])
                loss1 = criterion(output1, onehot, 1)
                loss2 = criterion(output2, onehot, 1)
                loss3 = criterion(output3, onehot, 1)

                loss4 = criterion(output4, onehot, 1)
                loss5 = criterion(output5, onehot, 1)
                loss6 = criterion(output6, onehot, 1)
                #mix up part#
                
            loss_all1 = 2*loss1[0]+loss1[1]+loss1[2]
            loss_all2 = 2*loss2[0]+loss2[1]+loss2[2]
            loss_all3 = 2*loss3[0]+loss3[1]+loss3[2]
            loss_all4 = 2*loss4[0]+loss4[1]+loss4[2]
            loss_all5 = 2*loss5[0]+loss5[1]+loss5[2]
            loss_all6 = 2*loss6[0]+loss6[1]+loss6[2]
            loss_all = 0.7*(loss_all1 + 0.8*loss_all2 + 0.8*loss_all3) + 0.3*(loss_all4 + 0.8*loss_all5 + 0.8*loss_all6)

            train_all_loss += loss_all

            if config.apex_flag:
                with amp.scale_loss(loss_all, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_all.backward()

            optimizer.step()
            optimizer.zero_grad()


            if i % 100 == 0:
                print(config.model_name + ' %0.7f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  %0.3f  | %s' % (\
                             rate, iter, epoch,
                             loss_all, loss1[0], loss1[1], loss1[2],
                             time_to_str((timer() - start),'min')))
            i += 1
            
        train_all_loss /= iter_per_epoch
        
        print('avg training loss: ', train_all_loss)
        log.write('Epoch: %1.0f LR: %0.6f avg training loss: %0.3f' % (epoch, rate, train_all_loss))
        log.write('\n')
        
        net.eval()
        val_loss_all, score_all, score1, score2, score3 = do_valid(net, valid_loader, loss_function)
        if max_score_print < score_all:
            max_score_print = score_all
        print('epoch: ', epoch, 'loss: ', val_loss_all, \
            'score: ', score_all, score1, score2, score3, 'max_score: ', max_score_print)
  
        log.write('valid: %0.3f  %0.3f  %0.3f  %0.3f  %0.3f' % (\
        val_loss_all, score_all, score1, score2, score3))
        log.write('\n')
        log.flush()
        
        if config.schedule == 'cosine':
            lr_scheduler.step()
        net.train()
            
        if max_score < score_all:
            max_score = score_all
            print('save max_score!!!!!! : ' + str(max_score))
            log.write('save max_score!!!!!! : ' + str(max_score))
            log.write('\n')
            log.flush()
            torch.save(net.state_dict(), out_dir + '/max_score_model.pth')

def main(config):
    if config.mode == 'train':
        run_train(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--meta_df', type=str, default='./train_with_fold.csv')
    parser.add_argument('--data_dir', type=str, default='/home/chec/data/bengali')
    parser.add_argument('--model', type=str, default='E5_aux_arc')
    parser.add_argument('--model_name', type=str, default='Enet_timm_aux_arc')
    parser.add_argument('--batch_size', type=int, default=440)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_h', type=int, default=137)#137 236
    parser.add_argument('--image_w', type=int, default=236)
    parser.add_argument('--beta', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train', choices=['train','test_classifier'])
    parser.add_argument('--pretrained_model', type=str, default='max_score_model.pth')

    parser.add_argument('--epoch_save_interval', type=int, default=10)
    parser.add_argument('--train_epoch', type=int, default=300)
    parser.add_argument('--apex_flag', type=bool, default=True)
    parser.add_argument('--base_lr', type=float, default=4e-4)
    parser.add_argument('--final_lr', type=float, default=1e-6)
    parser.add_argument('--schedule', type=str, default='normal', choices=['normal','cosine'])

    config = parser.parse_args()
    print(config)
    main(config)
