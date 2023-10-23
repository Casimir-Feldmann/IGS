import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import glob



masks_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/mask_color"
save_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/gt_frame_diff"
os.makedirs(save_dir, exist_ok=True)

masks_dir = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir))]

for idx in range(len(masks_dir)-1):

    mask1 = cv2.imread(masks_dir[idx], cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(masks_dir[idx+1], cv2.IMREAD_GRAYSCALE)

    mask1[mask1 > 0] = 1
    mask1[mask1 == 0] = 0

    mask2[mask2 > 0] = 1
    mask2[mask2 == 0] = 0

    diff = np.abs(mask2 - mask1) * 255
    # print(diff.shape)
    
    # plt.imshow(diff)
    # plt.show()


    cv2.imwrite(os.path.join(save_dir, masks_dir[idx].split('/')[-1]), diff)