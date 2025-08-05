# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Evaluating process                                            #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import math
import numpy as np

from config.config import TEST_CONFIG

class Evaluation():
    def __init__(self, dataset_config, threshold='loose'):
        self.dataset_config = dataset_config
        if threshold == 'loose':
            self.loc_thres = TEST_CONFIG['loose_loc_thres'] * dataset_config['ratio']
            self.ori_thres = TEST_CONFIG['loose_ori_thres']
        elif threshold == 'tight':
            self.loc_thres = TEST_CONFIG['tight_loc_thres'] * dataset_config['ratio']
            self.ori_thres = TEST_CONFIG['tight_ori_thres']
        else:
            raise ValueError(f'Unknown threshold: {threshold}')

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.loc_error = []
        self.ori_error = []
        self.type_correct = 0
        self.occ_correct = 0

    
    
    def distance(self, point1, point2):
        dist = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
        return dist

    def diff_angle(self, ori1, ori2):
        ori1 = ori1/np.linalg.norm(ori1)
        ori2 = ori2/np.linalg.norm(ori2)
        angle = np.arccos(max(min(1.0, np.dot(ori1, ori2)), -1.0))
        angle = angle*180/math.pi
        return angle



    def evaluate(self, final_slots, gt_slots):
        count = np.full(gt_slots.shape[0], False)
        while len(final_slots) > 0:
            check = False
            for k in range(len(gt_slots)):
                if (self.distance(final_slots[0][0:2], gt_slots[k][0:2]) < self.loc_thres) and (self.distance(final_slots[0][2:4], gt_slots[k][2:4]) < self.loc_thres):
                    if (self.diff_angle(final_slots[0][4:6], gt_slots[k][4:6]) < self.ori_thres) and (self.diff_angle(final_slots[0][6:8], gt_slots[k][6:8]) < self.ori_thres):
                        self.loc_error.append(self.distance(final_slots[0][0:2], gt_slots[k][0:2]))
                        self.loc_error.append(self.distance(final_slots[0][2:4], gt_slots[k][2:4]))
                        self.ori_error.append(self.diff_angle(final_slots[0][4:6], gt_slots[k][4:6]))
                        self.ori_error.append(self.diff_angle(final_slots[0][6:8], gt_slots[k][6:8]))

                        if final_slots[0][8] == gt_slots[k][8]:
                            self.type_correct += 1
                        if final_slots[0][9] == gt_slots[k][9]:
                            self.occ_correct += 1

                        self.TP += 1
                        final_slots = np.delete(final_slots, 0, 0)
                        count[k] = True
                        check = True
                        break

            if not check:
                self.FP += 1
                final_slots = np.delete(final_slots, 0, 0)
        
        for k in range(len(gt_slots)):
            if not count[k]:
                self.FN += 1



    def results(self):
        print(f'True positive: {self.TP}')
        print(f'False positive: {self.FP}')
        print(f'False negative: {self.FN}')

        recall = self.TP/(self.TP + self.FN)
        precision = self.TP/(self.TP + self.FP)

        print('\n')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')

        print('\n')
        print(f'Type classification rate: {self.type_correct/self.TP}')
        print(f'Occupancy classification rate: {self.occ_correct/self.TP}')

        print('\n')
        print(f'Mean loc: {np.mean(self.loc_error)}')
        print(f'Std loc: {np.std(self.loc_error)}')
        print(f'Mean ori: {np.mean(self.ori_error)}')
        print(f'Std ori: {np.std(self.ori_error)}')





