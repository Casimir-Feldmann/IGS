from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow

import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import cv2

config_file = '/home/casimir/ETH/SemesterProject/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
checkpoint_file = '/home/casimir/ETH/SemesterProject/mmflow/checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'

# init a model
model = init_model(config_file, checkpoint_file, device=device)

#Go up one dir and and change to that dir
os.chdir(os.path.dirname(os.getcwd()))

data_dir = "dataset_easy/stop_motion_3_processed"
save_dir = "dataset_easy/stop_motion_3_processed"

rgb_dirs = sorted(glob.glob(os.path.join(data_dir, 'rgb/*')))

flow_save_dir = os.path.join(save_dir, "flow")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(flow_save_dir, exist_ok=True)

for idx in tqdm(range(len(rgb_dirs))):
    
    if idx == len(rgb_dirs) - 1:
        break

    img1 = cv2.imread(rgb_dirs[idx])
    img2 = cv2.imread(rgb_dirs[idx+1])
    img1 = cv2.cvtColor(img1 ,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2 ,cv2.COLOR_BGR2RGB)

    result = inference_model(model, img1, img2)

    write_flow(result, flow_file=os.path.join(flow_save_dir, rgb_dirs[idx].split('/')[-1].replace("rgb", "flow").replace("png", "flo")))

    flow_map = visualize_flow(result, save_file=os.path.join(flow_save_dir, rgb_dirs[idx].split('/')[-1].replace("rgb", "flow")))

    # plt.imshow(flow_map)
    # plt.show()