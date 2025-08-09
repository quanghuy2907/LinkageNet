# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Training process                                              #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 

import torch
from torch.utils.data import DataLoader

from dataloader.data_pipeline import CustomDataset
from model.linkagenet import LinkageNet
from model.loss import Metric, Losses

from config.config import DATASET, TRAIN_CONFIG,MODEL_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CustomDataset(dataset=DATASET, state='train')
train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, drop_last=True)
val_dataset = CustomDataset(dataset=DATASET, state='val')
val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, drop_last=True)

model = LinkageNet(d_model=MODEL_CONFIG['d_model'])
model = model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr_init'], betas=[0.9, 0.999], eps=1e-8, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['epoch_num'], eta_min=TRAIN_CONFIG['lr_end'], last_epoch=-1)

# Create weights/log folder
if not os.path.isdir('./weights'):
    os.mkdir('./weights')
if not os.path.isdir('./log'):
    os.mkdir('./log')


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weight_folder = './weights/' + DATASET + '_' + timestamp
if not os.path.isdir(weight_folder):
    os.mkdir(weight_folder)
weight_path = weight_folder + '/model_weights.pth'

log_folder = './log/' + DATASET + '_' + timestamp
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
metric = Metric(log_folder)

loss = Losses(train_dataset.dataset_config)





best_loss = np.inf
start_time = time.time()
print('-----Start training-----')
for epoch in range(TRAIN_CONFIG['epoch_num']):
    print('EPOCH {}:'.format(epoch + 1))

    start = time.time()
    metric.reset()
    model.train()

    for i, data in enumerate(train_loader):
        imgs, labels_low, labels_high, _ = data
        imgs = imgs.to(device=device, dtype=torch.float32)
        labels_low = labels_low.to(device=device, dtype=torch.float32)
        labels_high = labels_high.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        g_output, l_output = model(imgs)

        t_losses = loss.compute(g_output, l_output, labels_low, labels_high)

        t_losses[-1].backward()
        optimizer.step()
        metric.update(t_losses)

        s = metric.compute()
        sys.stdout.write('\r' + f'step: {i}/{len(train_loader)} || {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} | {s[3]:.4f} | {s[4]:.4f} | {s[5]:.4f} | {s[6]:.4f} | {s[7]:.4f} | {s[8]:.4f} | {s[9]:.4f} | {s[10]:.4f} | {s[11]:.4f} | {s[12]:.4f} || {s[13]:.4f}')
    
    scheduler.step()
    metric.write(epoch, scheduler.get_last_lr()[0], 'train')
    
    print(f'\nTraining time for 1 epoch: {time.time() - start:.4f} secs')


    print('\nEvaluating')
    start = time.time()
    metric.reset()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            v_imgs, v_labels_low, v_labels_high, _ = data
            v_imgs = v_imgs.to(device=device, dtype=torch.float32)
            v_labels_low = v_labels_low.to(device=device, dtype=torch.float32)
            v_labels_high = v_labels_high.to(device=device, dtype=torch.float32)

            v_g_output, v_l_output = model(v_imgs)

            v_losses = loss.compute(v_g_output, v_l_output, v_labels_low, v_labels_high)

            metric.update(v_losses)

            s = metric.compute()
            sys.stdout.write('\r' + f'step: {i}/{len(val_loader)} || {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} | {s[3]:.4f} | {s[4]:.4f} | {s[5]:.4f} | {s[6]:.4f} | {s[7]:.4f} | {s[8]:.4f} | {s[9]:.4f} | {s[10]:.4f} | {s[11]:.4f} | {s[12]:.4f} || {s[13]:.4f}')
        
    metric.write(epoch, 0, 'val')

    if s[-1] < best_loss:
        print('\nTotal loss decreased from {} to {}, saving weights\n'.format(best_loss, s[-1]))
        best_loss = s[-1]
        torch.save(model.state_dict(), weight_path)
    else:
        print('\nTotal loss does not improve from {}'.format(best_loss))


    print(f'\nEvaluation time for 1 epoch: {time.time() - start:.4f} secs\n')


print(f'Total training time: {time.time() - start_time:.2f} secs\n')



