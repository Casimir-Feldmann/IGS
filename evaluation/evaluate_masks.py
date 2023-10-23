import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import glob
from evaluation_metrics import calc_ConfusionMatrix, calc_IoU_Sets

def eval_image(truth, pred, c=1):
    # Obtain predicted and actual condition
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt)
    tn = np.logical_and(not_pd, not_gt)
    fp = np.logical_and(pd, not_gt)
    fn = np.logical_and(not_pd, gt)

    result = np.zeros((gt.shape[0], gt.shape[1], 3)).astype('uint8')
    result[tp] = [255, 255, 255]
    result[tn] = [0, 0, 0]
    result[fp] = [255, 128, 0]
    result[fn] = [0, 0, 255]

    return result

gt_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/gt_frame_diff"
predicted_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/mask_flow"

gt_mask_dir = [os.path.join(gt_dir, file) for file in sorted(os.listdir(gt_dir))]
predicted_mask_dir = [os.path.join(predicted_dir, file) for file in sorted(os.listdir(predicted_dir))]

IoU_list = []

for gt_dir, pred_dir in zip(gt_mask_dir, predicted_mask_dir):
    
    pred_mask = cv2.imread(pred_dir, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

    pred_mask[pred_mask > 0] = 1
    pred_mask[pred_mask == 0] = 0

    gt_mask[gt_mask > 0] = 1
    gt_mask[gt_mask == 0] = 0

    #white tp 255 255 255
    #black tn 0 0 0
    #orange fp 255 128 0
    #blue fn 0 0 255
    
    visualized = eval_image(truth=gt_mask, pred=pred_mask)

    # plt.imshow(visualized)
    # plt.show()

    # tp, tn, fp, fn = calc_ConfusionMatrix(truth=gt_mask, pred=pred_mask)

    IoU = calc_IoU_Sets(truth=gt_mask, pred=pred_mask)
    IoU_list.append(IoU)

print("Average IoU:", sum(IoU_list)/len(IoU_list))

    

    





    
