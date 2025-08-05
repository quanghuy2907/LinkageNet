import numpy as np

import torch
from torch.utils.data import DataLoader

# from model.linkagenet import LinkageNet
# from model.backbone import Backbone
# from model.position_encoding import PositionEmbeddingSine
# from model.detr import DETR

# from model.matcher import HungarianMatcher

from dataloader.data_pipeline import CustomDataset

# from config import config
from inference.output_processing import OutputProcessing



# Test data loader
dataset = CustomDataset(dataset='ps2', state='train')
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

# count = 0
# for i, data in enumerate(data_loader):
#     images, labels_low, labels_high, slot_labels = data


#     count += torch.sum(labels_high[:, :, :, 6])
# a=1


out_tool = OutputProcessing(dataset.dataset_config, 'cpu')
slots = torch.tensor([[0.5, 1, 1.],
                      [0.9, 50, 50.],
                      [0.8, 45, 44.1],
                      [0.2, 50, 50.],
                      [0.5, 10, 1,],
                      [0.6, 100, 100]])

temp = out_tool.nms(slots)
a=1



# Test backbone
# backbone = Backbone(True)
# input = torch.rand(1, 3, 320, 320)
# feat = backbone(input)
# a=1

# Test model 
# model = LinkageNet(256).to('cuda')
# input = torch.rand(2, 3, 320, 320).to('cuda')
# out = model(input)
# a=1


# # Test position embedding
# pos_embed = PositionEmbeddingSine(num_pos_feats=62, normalize=True)
# pos = pos_embed(feat[0])


# a=1
# Test model
# model = DETR(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1,
#              num_encoder_layers=6, num_decoder_layers=6, 
#              num_queries=100, activation="relu", return_intermediate_dec=False).to('cuda')

# input = torch.rand(2, 3, 320, 320).to('cuda')
# out = model(input)

# a=1

# Test matcher
# preds = torch.tensor([[[1, 10, 10, 10, 10],
#                        [2, 1, 5, 1, 1],
#                        [3, 1.5, 1.1, 1, 5],
#                        [4, 1, 1, 10, 4]],
#                       [[1, 1, 4, 6, 2],
#                        [2, 11, 11, 11, 11],
#                        [3, 5, 1, 1, 1],
#                        [4, 1, 1, 5, 1]]], dtype=torch.float32)

# trues = torch.tensor([[[1, 1, 1, 1, 5],
#                        [0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0]],
#                       [[1, 1, 1, 5, 1],
#                        [1, 1, 3, 6, 3],
#                        [0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0]]], dtype=torch.float32)

# matcher = HungarianMatcher()
# indices = matcher(preds, trues)

# new_preds = preds[matcher.get_src_permutation_idx(indices)]
# new_trues = trues[matcher.get_tgt_permutation_idx(indices)]



a=1