import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from mmflow.datasets import visualize_flow, write_flow, read_flow

flows_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/flow"
save_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/mask_flow"
os.makedirs(save_dir, exist_ok=True)

flow_threshold = 2.0


for flow_file in sorted(os.listdir(flows_dir)):


    flow = read_flow(os.path.join(flows_dir, flow_file))
    # print(flow_file)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

    flow_mask = np.where(mag > flow_threshold, 1, 0)

    flow_mask = flow_mask * 255

    # plt.imshow(flow_mask)
    # plt.show()

    cv2.imwrite(os.path.join(save_dir, flow_file.replace("flow", "mask").replace(".flo", ".png")), flow_mask)