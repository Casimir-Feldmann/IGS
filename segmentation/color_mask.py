import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import mediapy as media
import os
import glob
from pathlib import Path
from tqdm import tqdm

def color_segment(img):
    low = np.array([100, 0, 0])
    high = np.array([255, 70, 70])

    mask = cv2.inRange(img, low, high)

    return mask


#Go up one dir and and change to that dir
os.chdir(os.path.dirname(os.getcwd()))

data_dir = "dataset/pairs_processed"
save_dir = "dataset/pairs_processed"

rgb_dirs = sorted(glob.glob(os.path.join(data_dir, 'rgb/*')))

mask_save_dir = os.path.join(save_dir, "mask_color")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

for image_dir in tqdm(rgb_dirs):

    sample_image = cv2.imread(image_dir)
    img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    mask = color_segment(img)

    # fg = cv2.bitwise_and(img, img, mask=mask)

    # save_img = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(mask_save_dir, image_dir.split('/')[-1].replace("rgb", "mask")), mask)



    


