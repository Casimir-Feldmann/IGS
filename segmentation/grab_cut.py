import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import glob


os.chdir(os.path.dirname(os.getcwd()))

data_dir = "dataset/pairs_processed"
save_dir = "dataset/pairs_processed"

rgb_dirs = sorted(glob.glob(os.path.join(data_dir, 'rgb/*')))
input_mask_dirs = sorted(glob.glob(os.path.join(data_dir, 'mask_color/*')))

mask_save_dir = os.path.join(save_dir, "mask_grabcut")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

for image_dir, mask_dir in tqdm(zip(rgb_dirs, input_mask_dirs)):


    img = cv2.imread(image_dir)
    assert img is not None, "file could not be read, check with os.path.exists()"

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    input_mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    input_mask[input_mask == 255] = 1

    # plt.imshow(mask)
    # plt.show()

    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.erode(input_mask, kernel, iterations=1) 

    # plt.imshow(mask_erosion)
    # plt.show()

    # mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (0,0,mask.shape[0],mask.shape[1])

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')

    # plt.imshow(mask2)
    # plt.show()
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img = img*mask2[:,:,np.newaxis]


    # plt.imshow(img),plt.colorbar(),plt.show()

    fg = cv2.bitwise_and(img, img, mask=mask2)
    bg = cv2.bitwise_and(img, img, mask=np.where(mask2==0, 1, 0).astype('uint8'))
    # plt.imshow(bg)
    # plt.show()
    cv2.imwrite("bg_grabcut.png", cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))

    bg = cv2.bitwise_and(img, img, mask=np.where(input_mask==0, 1, 0).astype('uint8'))
    # plt.imshow(bg)
    # plt.show()
    cv2.imwrite("bg_color_mask.png", cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))

    break



    cv2.imwrite(os.path.join(mask_save_dir, image_dir.split('/')[-1].replace("rgb", "mask")), mask2*255)