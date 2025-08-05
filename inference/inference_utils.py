# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Utils functions for inference                                 #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np



def max_in_3x3(conf, xid, yid):
    if xid > conf.shape[0] or yid > conf.shape[1] or xid < 0 or yid < 0:
        return 0, 0, 0
    
    # Pad the intinal array to make sure the 3x3 window can be defined
    conf = np.pad(conf, (1,1), mode='constant', constant_values=0)
    xid += 1
    yid += 1

    window = conf[xid-1:xid+2, yid-1:yid+2]

    # Get index of max value in the window
    flat_idx = np.argmax(window)

    # Convert falt index back to 2D offset
    offset = np.unravel_index(flat_idx, (3, 3))

    xid = (xid - 1) + offset[0]
    yid = (yid - 1) + offset[1]

    return conf[xid, yid], (xid - 1), (yid - 1)
