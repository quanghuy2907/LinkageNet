# ------------------------------------------------------------- #
# LinkageNet                                                    #
# LinkageNet model                                              #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class Backbone(nn.Module):
    def __init__(self, return_interm_layers: bool):
        super().__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

    def forward(self, input):
        xs = self.body(input)

        feat = []
        for name, x in xs.items():
            feat.append(x)

        return feat
    

class DetectionHead(nn.Module):
    def __init__(self, d_low_feat, d_high_feat, d_model):
        super().__init__()

        self.low_input_proj = nn.Conv2d(in_channels=d_low_feat, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.high_input_proj = nn.Conv2d(in_channels=d_high_feat, out_channels=d_model, kernel_size=1, stride=1, padding=0)

        # Global information
        self.g_center_conf = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.g_center_loc = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.g_entrance_len = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.g_entrance_ori = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.g_inside_slot = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.g_slot_type = nn.Conv2d(in_channels=d_model, out_channels=3, kernel_size=3, padding=1)
        self.g_slot_occ = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)

        # Local information
        self.l_right_junc_conf = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.l_left_junc_conf = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.l_junc_loc = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.l_junc_ori = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)
        self.l_sep_line_conf = nn.Conv2d(in_channels=d_model, out_channels=1, kernel_size=3, padding=1)
        self.l_sep_line_ori = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=3, padding=1)       


    def forward(self, low_feat, high_feat):
        low_src = self.low_input_proj(low_feat)
        center_conf = F.sigmoid(self.g_center_conf(low_src))
        center_loc = F.sigmoid(self.g_center_loc(low_src))
        entrance_len = F.sigmoid(self.g_entrance_len(low_src))
        entrance_ori = F.tanh(self.g_entrance_ori(low_src))
        inside_slot = F.sigmoid(self.g_inside_slot(low_src))
        slot_type = F.softmax(self.g_slot_type(low_src), dim=1)
        slot_occ = F.sigmoid(self.g_slot_occ(low_src))

        g_output = torch.cat([center_conf, center_loc, entrance_len, entrance_ori, inside_slot, slot_type, slot_occ], dim=1)

        high_src = self.high_input_proj(high_feat)
        right_junc_conf = F.sigmoid(self.l_right_junc_conf(high_src))
        left_junc_conf = F.sigmoid(self.l_left_junc_conf(high_src))
        junc_loc = F.sigmoid(self.l_junc_loc(high_src))
        junc_ori = F.tanh(self.l_junc_ori(high_src))
        sep_line_conf = F.sigmoid(self.l_sep_line_conf(high_src))
        sep_line_ori = F.tanh(self.l_sep_line_ori(high_src))

        l_output = torch.cat([right_junc_conf, left_junc_conf, junc_loc, junc_ori, sep_line_conf, sep_line_ori], dim=1)

        return g_output, l_output
    


class LinkageNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.backbone = Backbone(return_interm_layers=True)
        d_low_feat = self.backbone.num_channels[-1]
        d_high_feat = self.backbone.num_channels[-2]
        self.detection_head = DetectionHead(d_low_feat, d_high_feat, d_model)

    def forward(self, img_input):
        feats = self.backbone(img_input)

        g_output, l_output = self.detection_head(feats[-1], feats[-2])

        return g_output.permute(0, 2, 3, 1), l_output.permute(0, 2, 3, 1)