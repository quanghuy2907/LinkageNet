# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Process raw model output                                      #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F

from config.config import MODEL_CONFIG, TEST_CONFIG
from inference.inference_utils import *


class OutputProcessing():
    def __init__(self, dataset_config, device):
        self.dataset_config = dataset_config
        self.device = device


    # Non maximum suppression using entrance center
    #   Sort the entrance acording to confidence score
    #   Pick the entrance with highest confidence
    #   Delete others whose center is within 45 pixels with the selected one
    def nms(self, slots):
        new_slots = [slot for slot in slots]
        if len(new_slots) > 0:
            new_slots = sorted(new_slots, key=lambda x: x[0], reverse=True)

        nms_slots = []
        while new_slots:
            selected_slot = new_slots.pop(0)

            new_slots = [
                slot
                for slot in new_slots
                if torch.norm(torch.tensor((slot[1] - selected_slot[1], slot[2] - selected_slot[2]))) > TEST_CONFIG['nms_thres']
            ]

            nms_slots.append(selected_slot.numpy())

        return torch.tensor(np.array(nms_slots))
    


    # Calculating the slot prediction from model outputs
    def process_output(self, g_output, l_output):
        ## Get raw slot prediction from global information output
        h, w, c = g_output.shape
        raw_slots = g_output.reshape((-1, c))
        # Only take [center_conf, center_loc_x, center_loc_y, entrance_len, entrance_ori_x, entrance_ori_y]
        raw_slots = raw_slots[:, :6] 
        indices = torch.arange(len(raw_slots)).reshape(-1, 1)
        xid, yid = torch.unravel_index(indices, (h, w))
        
        # Remove predictions with low confidence 
        indices = raw_slots[:, 0] > TEST_CONFIG['center_thres']
        raw_slots = raw_slots[indices]
        xid = xid[indices].reshape(-1)
        yid = yid[indices].reshape(-1)
        
        # Change slot values to correct range
        if len(raw_slots) > 0:
            # Entrance center location
            raw_slots[:, 1] = ((raw_slots[:, 1] - 0.5) + (yid + 0.5)) * MODEL_CONFIG['low_feat_stride']
            raw_slots[:, 2] = ((raw_slots[:, 2] - 0.5) + (xid + 0.5)) * MODEL_CONFIG['low_feat_stride']
            # Entrance length
            raw_slots[:, 3] = raw_slots[:, 3] * self.dataset_config['l_max']
            # Normalize the entrance orientation
            raw_slots[:, 4:6] = F.normalize(raw_slots[:, 4:6], dim=-1)
        
        
        ## NMS
        nms_raw_slots = self.nms(raw_slots)
        # Change slot representation from entrance center to junction pair
        rough_slots = []
        for slot in nms_raw_slots:
            junc1 = [slot[1] - slot[4] * (slot[3] / 2),
                     slot[2] - slot[5] * (slot[3] / 2)]
            junc2 = [slot[1] + slot[4] * (slot[3] / 2),
                     slot[2] + slot[5] * (slot[3] / 2)]
            
            rough_slots.append([junc1[0], junc1[1], junc2[0], junc2[1]])


        ## Extract precise junctions' location and orientation from local information output
        precise_slots = []
        for slot in rough_slots:
            precise_slot = [-1, -1, -1, -1, -1, -1, -1, -1]
            for jid in range(2):
                # For each junction location: search the 3x3 window for the cell with highest confidence
                xid = int(slot[2 * jid + 1] / MODEL_CONFIG['high_feat_stride'])
                yid = int(slot[2 * jid] / MODEL_CONFIG['high_feat_stride'])
                max_conf, xid, yid = max_in_3x3(l_output[:, :, jid].numpy(), xid, yid)
                # Only take the junction with higher confidence value than threshold
                if max_conf < TEST_CONFIG['junc_thres']: break

                precise_slot[2 * jid] = ((yid + 0.5) + (l_output[xid, yid, 2] - 0.5)) * MODEL_CONFIG['high_feat_stride']
                precise_slot[2 * jid + 1] = ((xid + 0.5) + (l_output[xid, yid, 3] - 0.5)) * MODEL_CONFIG['high_feat_stride']
                
                # For each orientation: mean of the cell containing separating line
                rough_ori = F.normalize(l_output[xid, yid, 4:6], dim=-1)

                point = [precise_slot[2 * jid], precise_slot[2 * jid + 1]]
                orients = []
                for count in range(5):
                    if not ((0 <= point[0] < self.dataset_config['img_width']) and (0 <= point[1] < self.dataset_config['img_height'])): continue
                    yid = int(point[0] / MODEL_CONFIG['high_feat_stride'])
                    xid = int(point[1] / MODEL_CONFIG['high_feat_stride'])

                    if l_output[xid, yid, 6] > TEST_CONFIG['sep_line_thres']:
                        orients.append([l_output[xid, yid, 7], l_output[xid, yid, 8]])

                    point = [point[0] + 8 * rough_ori[0],
                             point[1] + 8 * rough_ori[1]]

                if len(orients) == 0:
                    precise_slot[4 + 2 * jid] = rough_ori[0]
                    precise_slot[4 + 2 * jid + 1] = rough_ori[1]
                else:
                    precise_ori = F.normalize(torch.mean(torch.tensor(orients), dim=0), dim=-1)

                    precise_slot[4 + 2 * jid] = precise_ori[0]
                    precise_slot[4 + 2 * jid + 1] = precise_ori[1]

            if -1 not in precise_slot:
                precise_slots.append(precise_slot)

        
        ## Extract slot type and occupancy
        final_slots = []
        for slot in precise_slots:
            slot_center = [(slot[0] + slot[2]) / 2 + 16 * slot[4],
                           (slot[1] + slot[3]) / 2 + 16 * slot[5]]
            xid = int(slot_center[1] / MODEL_CONFIG['low_feat_stride'])
            yid = int(slot_center[0] / MODEL_CONFIG['low_feat_stride'])

            max_conf, xid, yid = max_in_3x3(g_output[:, :, 6].numpy(), xid, yid)

            slot_type = torch.argmax(g_output[xid, yid, 7:10], dim=-1)
            slot_occ = g_output[xid, yid, 10]
            if slot_occ > 0.5:
                slot_occ = 1
            else:
                slot_occ = 0

            final_slots.append([slot[0].item(), slot[1].item(), slot[2].item(), slot[3].item(), slot[4].item(), slot[5].item(), slot[6].item(), slot[7].item(), slot_type.item(), slot_occ])
    
        return np.array(final_slots)
            

    

    def visualize_preds(self, imgs, final_slots, index):
        img = imgs[index].detach().cpu().permute(1,2,0)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)

        for item in final_slots:
            junc1 = np.array([item[0], item[1]])
            junc2 = np.array([item[2], item[3]])

            ori1 = np.array([junc1[0] + item[4]*25, junc1[1] + item[5]*25])
            ori2 = np.array([junc2[0] + item[6]*25, junc2[1] + item[7]*25])

            slot_type = int(item[8])
            slot_occ = item[9]
            if slot_occ == 0:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
            else:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
        
        # plt.savefig('b.png')
        # plt.close()
        plt.show()


    