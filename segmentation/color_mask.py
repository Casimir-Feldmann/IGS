import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mediapy as media
import os
import glob
from pathlib import Path
from tqdm import tqdm

data_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/pairs_processed"
save_dir = "/home/casimir/ETH/SemesterProject/IGS/dataset_easy/pairs_processed"

rgb_dirs = glob.glob(os.path.join(data_dir, 'rgb/*'))

mask_save_dir = os.path.join(save_dir, "mask")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

for image_dir in tqdm(rgb_dirs):

    sample_image = cv2.imread(image_dir)
    img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

    low = np.array([100, 0, 0])
    high = np.array([255, 70, 70])

    mask = cv2.inRange(img, low, high)

    # result = cv2.bitwise_and(img, img, mask=mask)

    # save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(mask_save_dir, image_dir.split('/')[-1].replace("rgb", "mask")), mask)



    


