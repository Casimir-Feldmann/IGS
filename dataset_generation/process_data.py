import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mediapy as media
import os
import glob
from pathlib import Path
from tqdm import tqdm
import os


top_left_coords = [322,66]
bottom_right_coords = [1130,613]

#Go up one dir and and change to that dir
os.chdir(os.path.dirname(os.getcwd()))

data_dir = "dataset/pairs"
save_dir = "dataset/pairs_processed"

rgb_dirs = glob.glob(os.path.join(data_dir, 'rgb/*'))
depth_dirs = glob.glob(os.path.join(data_dir, 'depth/*'))

rgb_save_dir = os.path.join(save_dir, "rgb")
depth_save_dir = os.path.join(save_dir, "depth")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(rgb_save_dir, exist_ok=True)
os.makedirs(depth_save_dir, exist_ok=True)


for image_dir in tqdm(rgb_dirs):

    img = cv2.imread(image_dir)
    # img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    crop_img = img[top_left_coords[1]:bottom_right_coords[1], top_left_coords[0]:bottom_right_coords[0],:]

    # output = Image.fromarray(crop_img)
    

    # output.save(os.path.join(rgb_save_dir, image_dir.split('/')[-1]))

    cv2.imwrite(os.path.join(rgb_save_dir, image_dir.split('/')[-1]), crop_img)

    


