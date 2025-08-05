# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Loss funtions                                                 #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: May 2025                                       #
# ------------------------------------------------------------- #


import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.aggregation import MeanMetric

from config.config import LOSS_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Metric():
    def __init__(self, log_folder):
        self.g_center_conf = MeanMetric().to(device)
        self.g_center_loc = MeanMetric().to(device)
        self.g_entrance_len = MeanMetric().to(device)
        self.g_entrance_ori = MeanMetric().to(device)
        self.g_inside_slot = MeanMetric().to(device)
        self.g_slot_type = MeanMetric().to(device)
        self.g_slot_occ = MeanMetric().to(device)
        self.l_right_junc_conf = MeanMetric().to(device)
        self.l_left_junc_conf = MeanMetric().to(device)
        self.l_junc_loc = MeanMetric().to(device)
        self.l_junc_ori = MeanMetric().to(device)
        self.l_sep_line_conf = MeanMetric().to(device)
        self.l_sep_line_ori = MeanMetric().to(device)
        self.total_loss = MeanMetric().to(device)

        self.writer = SummaryWriter(log_folder)
    
    def reset(self):
        self.g_center_conf.reset()
        self.g_center_loc.reset()
        self.g_entrance_len.reset()
        self.g_entrance_ori.reset()
        self.g_inside_slot.reset()
        self.g_slot_type.reset()
        self.g_slot_occ.reset()
        self.l_right_junc_conf.reset()
        self.l_left_junc_conf.reset()
        self.l_junc_loc.reset()
        self.l_junc_ori.reset()
        self.l_sep_line_conf.reset()
        self.l_sep_line_ori.reset()
        self.total_loss.reset()

    def update(self, losses):
        self.g_center_conf.update(losses[0])
        self.g_center_loc.update(losses[1])
        self.g_entrance_len.update(losses[2])
        self.g_entrance_ori.update(losses[3])
        self.g_inside_slot.update(losses[4])
        self.g_slot_type.update(losses[5])
        self.g_slot_occ.update(losses[6])
        self.l_right_junc_conf.update(losses[7])
        self.l_left_junc_conf.update(losses[8])
        self.l_junc_loc.update(losses[9])
        self.l_junc_ori.update(losses[10])
        self.l_sep_line_conf.update(losses[11])
        self.l_sep_line_ori.update(losses[12])
        self.total_loss.update(losses[13])
    
    def write(self, epoch, lr, state):
        if state == 'train':
            self.writer.add_scalar('lr', lr, global_step=epoch)

        self.writer.add_scalar(state + '_loss/center_conf_loss', self.g_center_conf.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/center_loc_loss', self.g_center_loc.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/entrance_len_loss', self.g_entrance_len.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/entrance_ori_loss', self.g_entrance_ori.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/inside_slot_loss', self.g_inside_slot.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/slot_type_loss', self.g_slot_type.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/slot_occ_loss', self.g_slot_occ.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/right_junc_conf_loss', self.l_right_junc_conf.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/left_junc_conf_loss', self.l_left_junc_conf.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/junc_loc_loss', self.l_junc_loc.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/junc_ori_loss', self.l_junc_ori.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/sep_line_conf_loss', self.l_sep_line_conf.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/sep_line_ori_loss', self.l_sep_line_ori.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/total_loss', self.total_loss.compute(), global_step=epoch)

        self.writer.flush()

    def compute(self):
        return [self.g_center_conf.compute(),
                self.g_center_loc.compute(),
                self.g_entrance_len.compute(),
                self.g_entrance_ori.compute(),
                self.g_inside_slot.compute(),
                self.g_slot_type.compute(),
                self.g_slot_occ.compute(),
                self.l_right_junc_conf.compute(),
                self.l_left_junc_conf.compute(),
                self.l_junc_loc.compute(),
                self.l_junc_ori.compute(),
                self.l_sep_line_conf.compute(),
                self.l_sep_line_ori.compute(),
                self.total_loss.compute()]
        





class Losses():
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config


    def conf_loss(self, pred, true, weight):
        loss = torch.square(pred - true)
        loss = (true == 0) * loss * weight[0] + (true == 1) * loss * weight[1]
        # loss = torch.sum(loss, dim=[1, 2])
        loss = torch.mean(loss) 

        return loss
    
    def loc_loss(self, pred, true, mask):
        loss = torch.square(pred - true)
        loss = mask * torch.sum(loss, -1)
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def ori_loss(self, pred, true, mask):
        pred = F.normalize(pred, dim=-1)
        true = F.normalize(true, dim=-1)

        loss = torch.square(pred - true)
        loss = mask * torch.sum(loss, -1)
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss

    def len_loss(self, pred, true, mask):
        loss = torch.square(pred - true)
        loss = mask * loss
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def type_loss(self, pred, true, mask, weight):
        pred = pred.permute(0, 3, 1, 2)
        true = torch.argmax(true, dim=-1).to(torch.long)#.permute(0, 3, 1, 2)
        class_weights = torch.tensor(weight).to(pred.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        loss = loss_fn(pred, true)
        loss = mask * loss
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def occ_loss(self, pred, true, mask, weight):
        loss = torch.square(pred - true)
        loss = mask * loss
        loss = (true == 0) * loss * weight[0] + (true == 1) * loss * weight[1]
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss



    def compute(self, g_output, l_output, labels_low, labels_high):
        center_conf = self.conf_loss(g_output[:, :, :, 0], labels_low[:, :, :, 0], self.dataset_config['center_weights']) * LOSS_CONFIG['center_conf']
        center_loc = self.loc_loss(g_output[:, :, :, 1:3] - 0.5, labels_low[:, :, :, 1:3], labels_low[:, :, :, 0] == 1) * LOSS_CONFIG['center_loc']
        entrance_len = self.len_loss(g_output[:, :, :, 3], labels_low[:, :, :, 3], labels_low[:, :, :, 0] == 1) * LOSS_CONFIG['entrance_len']
        entrance_ori = self.ori_loss(g_output[:, :, :, 4:6], labels_low[:, :, :, 4:6], labels_low[:, :, :, 0] == 1) * LOSS_CONFIG['entrance_ori']
        inside_slot = self.conf_loss(g_output[:, :, :, 6], labels_low[:, :, :, 6], self.dataset_config['inslot_weights']) * LOSS_CONFIG['inside_slot']
        slot_type = self.type_loss(g_output[:, :, :, 7:10], labels_low[:, :, :, 7:10], labels_low[:, :, :, 6] == 1, self.dataset_config['class_weights']) * LOSS_CONFIG['slot_type']
        slot_occ = self.occ_loss(g_output[:, :, :, 10], labels_low[:, :, :, 10], labels_low[:, :, :, 6] == 1, self.dataset_config['occupancy_weights']) * LOSS_CONFIG['slot_occ']

        right_junc_conf = self.conf_loss(l_output[:, :, :, 0], labels_high[:, :, :, 0], self.dataset_config['junc_weights']) * LOSS_CONFIG['junc_conf']
        left_junc_conf = self.conf_loss(l_output[:, :, :, 1], labels_high[:, :, :, 1], self.dataset_config['junc_weights']) * LOSS_CONFIG['junc_conf']
        junc_loc = self.loc_loss(l_output[:, :, :, 2:4] - 0.5, labels_high[:, :, :, 2:4], torch.logical_or(labels_high[:, :, :, 0] == 1, labels_high[:, :, :, 1] == 1)) * LOSS_CONFIG['junc_loc']
        junc_ori = self.ori_loss(l_output[:, :, :, 4:6], labels_high[:, :, :, 4:6], torch.logical_or(labels_high[:, :, :, 0] == 1, labels_high[:, :, :, 1] == 1)) * LOSS_CONFIG['junc_ori']
        sep_line_conf = self.conf_loss(l_output[:, :, :, 6], labels_high[:, :, :, 6], self.dataset_config['sep_line_weights']) * LOSS_CONFIG['sep_line_conf']
        sep_line_ori = self.ori_loss(l_output[:, :, :, 7:9], labels_high[:, :, :, 7:9], labels_high[:, :, :, 6] == 1) * LOSS_CONFIG['sep_line_ori']

        total = center_conf + center_loc + entrance_len + entrance_ori + inside_slot + slot_type + slot_occ + right_junc_conf + left_junc_conf + junc_loc + junc_ori + sep_line_conf + sep_line_ori

        return [center_conf, center_loc, entrance_len, entrance_ori, inside_slot, slot_type, slot_occ, right_junc_conf, left_junc_conf, junc_loc, junc_ori, sep_line_conf, sep_line_ori, total]


