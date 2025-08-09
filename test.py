# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Testing process                                               #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
import time

import torch
from torch.utils.data import DataLoader

from dataloader.data_pipeline import CustomDataset
from model.linkagenet import LinkageNet
from config.config import DATASET, MODEL_CONFIG, TEST_CONFIG
from inference.output_processing import OutputProcessing
from inference.evaluation import Evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = CustomDataset(dataset=DATASET, state='test')
test_loader = DataLoader(test_dataset, batch_size=TEST_CONFIG['batch_size'], shuffle=False, drop_last=False)

model = LinkageNet(d_model=MODEL_CONFIG['d_model'])
model = model.to(device=device)

model.load_state_dict(torch.load(TEST_CONFIG['weight_path', weights_only=True))
model.eval()

output_processing_tool = OutputProcessing(test_dataset.dataset_config, device)
eval_tool = Evaluation(test_dataset.dataset_config, threshold=TEST_CONFIG['evaluation_mode'])

inf_time = 0

start_time = time.time()
print('-----Start testing-----')
for idx, data in enumerate(test_loader):
    imgs, _, _, labels = data
    imgs = imgs.to(device=device, dtype=torch.float32)

    t1 = time.time()
    g_output, l_output = model(imgs)
    inf_time += (time.time() - t1)

    for i in range(TEST_CONFIG['batch_size']):
        sys.stdout.write('\r' + 'idx=' + str(i + idx*TEST_CONFIG['batch_size'] + 1))

        final_slots = output_processing_tool.process_output(g_output[i].detach().cpu(), l_output[i].detach().cpu())        
        gt_slots = labels[i][labels[i, :, 0] != 0]
        eval_tool.evaluate(final_slots, gt_slots)


print(f'\nTotal testing time: {time.time() - start_time:.4f} secs')

inference_time = inf_time / (len(test_loader) * TEST_CONFIG['batch_size'])
print(f'Inference time: {inference_time} sec/frame')
print(f'Inference time: {1/ inference_time} fps')
print('\n')

eval_tool.results()
