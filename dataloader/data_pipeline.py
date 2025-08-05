# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Input pipeline                                                #
#       - Read images                                           #
#       - Read annotations                                      #
#       - Data augmentation                                     #
#       - Generate labels                                       #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import os
import numpy as np
import skimage
import scipy
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config.config import DATASET_CONFIG, MODEL_CONFIG
from dataloader.dataloader_utils import inside_slot

class CustomDataset(Dataset):
    def __init__(self, dataset='ps2', state='train'):  
        if dataset not in DATASET_CONFIG:
            raise ValueError(f'Unknown dataset: {dataset}')
        
        self.dataset = dataset
        self.dataset_config = DATASET_CONFIG[dataset]
        self.dataset_config['low_grid_height'] = int(self.dataset_config['img_height'] / MODEL_CONFIG['low_feat_stride'])
        self.dataset_config['low_grid_width'] = int(self.dataset_config['img_width'] / MODEL_CONFIG['low_feat_stride'])
        self.dataset_config['high_grid_height'] = int(self.dataset_config['img_height'] / MODEL_CONFIG['high_feat_stride'])
        self.dataset_config['high_grid_width'] = int(self.dataset_config['img_width'] / MODEL_CONFIG['high_feat_stride'])

        self.img_path = os.path.join(self.dataset_config['dataset_path'], 'images', state if state == 'train' else 'test')
        self.label_path = os.path.join(self.dataset_config['dataset_path'], 'labels', state if state == 'train' else 'test')
        self.img_files = os.listdir(self.img_path)
        self.img_files.sort()

        self.state = state



    def __len__(self):
        return len(self.img_files)
    


    # Extract slot information from raw .mat data file 
    def read_slot_annotation(self, file_path):
        data = scipy.io.loadmat(file_path)

        temp_junctions = []
        temp_slots = []
        slots = []

        if self.dataset == 'ps2':
            for i in range(len(data['marks'])):
                temp_junctions.append([float(data['marks'][i][0]), float(data['marks'][i][1]), 
                                    float(data['marks'][i][2]), float(data['marks'][i][3])])
            for i in range(len(data['slots'])):  
                temp_slots.append([int(data['slots'][i][0]), int(data['slots'][i][1]), 
                                int(data['slots'][i][7] - 1),
                                int(data['slots'][i][8])])
            for slot in temp_slots:
                slot_type = slot[2]
                slot_occ = slot[3]

                point1 = [temp_junctions[int(slot[0] - 1)][0], temp_junctions[int(slot[0] - 1)][1]]
                ori1 = [temp_junctions[int(slot[0] - 1)][2], temp_junctions[int(slot[0] - 1)][3]]
                
                point2 = [temp_junctions[int(slot[1] - 1)][0], temp_junctions[int(slot[1] - 1)][1]]
                ori2 = [temp_junctions[int(slot[1] - 1)][2], temp_junctions[int(slot[1] - 1)][3]]

                slots.append([point1[0], point1[1], point2[0], point2[1],
                              ori1[0], ori1[1], ori2[0], ori2[1], 
                              slot_type, slot_occ])
        else: # SNU
            for i in range(len(data['marks'])):
                temp_junctions.append([float(data['marks'][i][0]), float(data['marks'][i][1])])
            for i in range(len(data['slots'])):  
                temp_slots.append([int(data['slots'][i][0]), int(data['slots'][i][1]), 
                                int(data['slots'][i][2]), int(data['slots'][i][3]),
                                int(data['slots'][i][4]),
                                int(data['slots'][i][5])])
            for slot in temp_slots:
                slot_type = slot[4]
                slot_occ = slot[5]

                point1 = np.array([temp_junctions[int(slot[0] - 1)][0], temp_junctions[int(slot[0] - 1)][1]])
                point2 = np.array([temp_junctions[int(slot[1] - 1)][0], temp_junctions[int(slot[1] - 1)][1]])
                point3 = np.array([temp_junctions[int(slot[2] - 1)][0], temp_junctions[int(slot[2] - 1)][1]])
                point4 = np.array([temp_junctions[int(slot[3] - 1)][0], temp_junctions[int(slot[3] - 1)][1]])

                ori1 = point4 - point1
                ori1 = ori1 / (np.linalg.norm(ori1) + 1e-9)
                ori2 = point3 - point2
                ori2 = ori2 / (np.linalg.norm(ori2) + 1e-9)
                if np.linalg.norm(ori1) == 0:
                    ori1 = ori2
                if np.linalg.norm(ori2) == 0:
                    ori2 = ori1
                if np.linalg.norm(ori1) == 0 and np.linalg.norm(ori2) == 0:
                    print(f'There is a slot with error label in {file_path}')
                    continue

                slots.append([point1[0], point1[1], point2[0], point2[1],
                              ori1[0], ori1[1], ori2[0], ori2[1],
                              slot_type, slot_occ])

        slots = np.array(slots)
        if len(slots) > 0:
            slots[:, 0:4] *= self.dataset_config['ratio']

        return slots
    


    # Implement aumentation for the input (include horizontal and vertical flip)
    def augment(self, img, slots):
        aug_img = img.copy()
        aug_slots = slots.copy()

        # Vertical flip
        if np.random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 0)
            temp = []
            for slot in aug_slots:
                temp.append([slot[2], self.dataset_config['img_height'] - slot[3],
                             slot[0], self.dataset_config['img_height'] - slot[1],
                             slot[6], -slot[7],
                             slot[4], -slot[5],
                             slot[8],
                             slot[9]])
            aug_slots = np.array(temp)
        
        # Horizontal flip (only work for PS2.0 dataset)
        if self.dataset_config['flip_horizontal']:
            if np.random.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
                temp = []
                for slot in aug_slots:
                    temp.append([self.dataset_config['img_width'] - slot[2], slot[3],
                                 self.dataset_config['img_width'] - slot[0], slot[1],
                                 -slot[6], slot[7],
                                 -slot[4], slot[5],
                                 slot[8],
                                 slot[9]])
                aug_slots = np.array(temp)
            
        return aug_img, aug_slots



    # Generate label for training
    def generate_slot_labels(self, slots, state):
        if state == 'train' or state == 'val':
            labels_low = np.zeros((self.dataset_config['low_grid_height'], self.dataset_config['low_grid_width'], MODEL_CONFIG['low_label_channel']))
            labels_high = np.zeros((self.dataset_config['high_grid_height'], self.dataset_config['high_grid_width'], MODEL_CONFIG['high_label_channel']))

            for slot in slots:
                junction1 = slot[0:2]
                junction2 = slot[2:4]
                ori1 = slot[4:6]
                ori2 = slot[6:8]
                slot_type = int(slot[8])
                slot_occ = slot[9]

                # Labels for low feature map
                # Posibility that the cell contains the entrance center
                entrance_center = (junction1 + junction2) / 2
                xid = int(entrance_center[1] / MODEL_CONFIG['low_feat_stride'])
                yid = int(entrance_center[0] / MODEL_CONFIG['low_feat_stride'])
                labels_low[xid, yid, 0] = 1
                # Entrance center location (relative position to cell center, value from -0.5 to 0.5)
                labels_low[xid, yid, 1] = entrance_center[0] / MODEL_CONFIG['low_feat_stride'] - (yid + 0.5)
                labels_low[xid, yid, 2] = entrance_center[1] / MODEL_CONFIG['low_feat_stride'] - (xid + 0.5)
                # Length of the slot entrance
                labels_low[xid, yid, 3] = np.linalg.norm(junction1 - junction2) / self.dataset_config['l_max']
                # Orientation of the slot entrance
                entrance_ori = junction2 - junction1
                entrance_ori = entrance_ori / np.linalg.norm(entrance_ori)
                labels_low[xid, yid, 4:6] = entrance_ori

                # Fill data for cells inside parking slot
                for cxid in range(self.dataset_config['low_grid_height']):
                    for cyid in range(self.dataset_config['low_grid_width']):
                        check = inside_slot((cyid, cxid), slot, self.dataset_config['img_height'], self.dataset_config['img_width'], self.dataset_config['l_mean'])
                        if (check == 1) or ((cxid == xid) and (cyid == yid)): # Cells containing entrance center are also considered as inside that parking slot
                            # Posibility that the cell in inside a parking slot
                            labels_low[cxid, cyid, 6] = 1
                            # Type of the parking slot
                            labels_low[cxid, cyid, 7 + slot_type] = 1
                            # Occupancy of the parking slot
                            labels_low[cxid, cyid, 10] = slot_occ
                        elif check == -1:
                            labels_low[cxid, cyid, 6] = -1
                

                # Labels for high feature map
                for id, (junction, ori) in enumerate([[junction1, ori1], [junction2, ori2]]):
                    jxid = int(junction[1] / MODEL_CONFIG['high_feat_stride'])
                    jyid = int(junction[0] / MODEL_CONFIG['high_feat_stride'])
                    # In case the junction lays on the image edges
                    if jxid == self.dataset_config['high_grid_height']:
                        jxid -= 1
                    if jyid == self.dataset_config['high_grid_width']:
                        jyid -= 1
                    
                    to_fill = (labels_high[jxid, jyid, 0] == 0) and (labels_high[jxid, jyid, 1] == 0)
                    # Possibility that the cell contains a junction (id=0 is left junction, id=1 is right junction)
                    labels_high[jxid, jyid, id] = 1

                    if to_fill:
                        # Junction location (relative position to cell center, value from -0.5 to 0.5)
                        labels_high[jxid, jyid, 2] = junction[0] / MODEL_CONFIG['high_feat_stride'] - (jyid + 0.5)
                        labels_high[jxid, jyid, 3] = junction[1] / MODEL_CONFIG['high_feat_stride'] - (jxid + 0.5)
                        # Junction orientation
                        labels_high[jxid, jyid, 4:6] = ori

                        # Fill data for cells containing separating line
                        length = self.dataset_config['l_mean'][slot_type]
                        point = junction.copy()
                        count = 0
                        while ((0 <= point[0] < self.dataset_config['img_width']) and (0 <= point[1] < self.dataset_config['img_height'])) and np.linalg.norm(junction - point) < length:
                            count += 1
                            oxid = int(point[1] / MODEL_CONFIG['high_feat_stride'])
                            oyid = int(point[0] / MODEL_CONFIG['high_feat_stride'])
                            if count <= 20:
                                # Posibility that the cell contains parts of a separating line
                                labels_high[oxid, oyid, 6] = 1
                                # Orientation of the separating line
                                labels_high[oxid, oyid, 7:9] = ori
                            # else:
                            #     # Only 10 cells are considered, other cells at further positions are ignored
                            #     labels_high[oxid, oyid, 6] = -1

                            point = [point[0] + 8 * ori[0],
                                     point[1] + 8 * ori[1]]
                            
            return labels_low, labels_high, 0
        
        elif state == 'test':
            slot_labels = slots
            if len(slot_labels) != 0:
                slot_labels = np.pad(slot_labels, pad_width=((0, 10 - len(slots)), (0, 0)), mode='constant', constant_values=0)
            else:
                slot_labels = np.zeros((10, 10))
            
            return 0, 0, slot_labels



    # Visualization function, for testing
    def visualize_label(self, imgs, labels_low, labels_high, slot_labels, index, mode=0):
        plt.figure(figsize=(10, 10))
        plt.axis('off')

        img = imgs[index].detach().cpu().permute(1, 2, 0)
        plt.imshow(img)


        if torch.sum(labels_low) != 0:
            label_low = labels_low[index].detach().cpu().numpy()
            label_high = labels_high[index].detach().cpu().numpy()
        if torch.sum(slot_labels) != 0:
            slot_label = slot_labels[index].detach().cpu().numpy()
            slot_label = slot_label[slot_label[:, 0] != 0]

        if mode == 0: # full visualization
            if torch.sum(slot_labels) != 0:
                for slot in slot_label:
                    junc1 = slot[0:2]
                    junc2 = slot[2:4]
                    ori1 = slot[4:6]
                    ori1 = junc1 + ori1 * 25
                    ori2 = slot[6:8]
                    ori2 = junc2 + ori2 * 25

                    slot_type = int(slot[8])
                    slot_occ = slot[9]

                    color = self.dataset_config['color_dict'][slot_type]
                    if slot_occ == 0.0:
                        plt.plot(junc1[0], junc1[1], 'o', color=color, markersize=12)
                        plt.plot(junc2[0], junc2[1], 'o', color=color, markersize=12)
                        plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=color, linewidth=3)
                        plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=color, linewidth=3)
                        plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=color, linewidth=3)
                    else:
                        plt.plot(junc1[0], junc1[1], 'o', color=color, markersize=12)
                        plt.plot(junc2[0], junc2[1], 'o', color=color, markersize=12)
                        plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=color, linewidth=3)
                        plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=color, linewidth=3)
                        plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=color, linewidth=3)
            elif torch.sum(labels_low) != 0:
                label = label_low.reshape([-1, MODEL_CONFIG['low_label_channel']])
                for id, item in enumerate(label):
                    if item[0] != 0:
                        xid, yid = np.unravel_index(id, (self.dataset_config['low_grid_height'], self.dataset_config['low_grid_width']))
                        center = np.array([item[1] + (yid + 0.5),
                                           item[2] + (xid + 0.5)]) * MODEL_CONFIG['low_feat_stride']
                        
                        junc1 = np.array([center[0] - item[4] * (item[3] * self.dataset_config['l_max'] / 2),
                                          center[1] - item[5] * (item[3] * self.dataset_config['l_max'] / 2)])
                        junc2 = np.array([center[0] + item[4] * (item[3] * self.dataset_config['l_max'] / 2),
                                          center[1] + item[5] * (item[3] * self.dataset_config['l_max'] / 2)])
                        
                        slot_type = np.argmax(item[7:10])
                        slot_occ = item[10]

                        color = self.dataset_config['color_dict'][slot_type]
                        if slot_occ == 0.0:
                            plt.plot(junc1[0], junc1[1], 'o', color=color, markersize=12)
                            plt.plot(junc2[0], junc2[1], 'o', color=color, markersize=12)
                            plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=color, linewidth=3)
                        else:
                            plt.plot(junc1[0], junc1[1], 'o', color=color, markersize=12)
                            plt.plot(junc2[0], junc2[1], 'o', color=color, markersize=12)
                            plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=color, linewidth=3)

        elif mode == 1: # inside slot visualization
            if torch.sum(labels_low) != 0:
                for xid in range(self.dataset_config['low_grid_height']):
                    for yid in range(self.dataset_config['low_grid_width']):
                        slot_type = np.argmax(label_low[xid, yid, 7:10])
                        if label_low[xid, yid, 6] == 1:
                            c = self.dataset_config['color_dict'][slot_type]
                        elif label_low[xid, yid, 6] == 0:
                            c = 'k'
                        elif label_low[xid, yid, 6] == -1:
                            c = 'k'
                        
                        plt.plot((yid + 0.5) * MODEL_CONFIG['low_feat_stride'], (xid + 0.5) * MODEL_CONFIG['low_feat_stride'], 'o', color=c, markersize=12)

        elif mode == 2: # labels_high visualization
            if torch.sum(labels_high) != 0:
                for xid in range(self.dataset_config['high_grid_height']):
                    for yid in range(self.dataset_config['high_grid_width']):
                        junc = np.array([label_high[xid, yid, 2] + (yid + 0.5),
                                         label_high[xid, yid, 3] + (xid + 0.5)]) * MODEL_CONFIG['high_feat_stride']
                        if label_high[xid, yid, 0] == 1:
                            plt.plot(junc[0], junc[1], 'o', color='cyan', markersize=16)
                        if label_high[xid, yid, 1] == 1:
                            plt.plot(junc[0], junc[1], 'o', color='orange', markersize=12)

                        if label_high[xid, yid, 6] == 1:
                            p1 = np.array([yid + 0.5, xid + 0.5]) * MODEL_CONFIG['high_feat_stride']
                            p2 = p1 + label_high[xid, yid, 7:9] * 10

                            plt.plot(p1[0], p1[1], 'o', color='magenta', markersize=8)
                            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='magenta', linewidth=3)


        plt.show()


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_files[idx])
        image = skimage.io.imread(img_path)
        image = np.uint8(np.round(skimage.transform.resize(image, (self.dataset_config['img_height'], self.dataset_config['img_width']), preserve_range=True)))

        label_path = os.path.join(self.label_path, self.img_files[idx].replace('jpg', 'mat'))
        slots = self.read_slot_annotation(label_path)

        # Augmentation (flip horizontal + vertical)
        if self.state == 'train':
            aug_image, aug_slots = self.augment(image, slots)
        else:
            aug_image = image.copy()
            aug_slots = slots.copy()

        aug_image = transforms.ToTensor()(aug_image)
        labels_low, labels_high, slot_labels = self.generate_slot_labels(aug_slots, self.state)

        return aug_image, labels_low, labels_high, slot_labels
    