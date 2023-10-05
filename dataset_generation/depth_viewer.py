import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mediapy as media

MIN_DEPTH = 0.4
MAX_DEPTH = 0.8

gt = Image.open('/home/casimir/ETH/SemesterProject/dataset/pairs/depth/00131_depth.png')


depth = np.asarray(gt, dtype=np.float32)
depth = depth * 0.001


depth = np.where(depth > MIN_DEPTH, depth, 0)
depth = np.where(depth < MAX_DEPTH, depth, 0)


print(depth.min())
print(depth.max())


plt.imshow(depth)
plt.show()


rgb_depth_image = media.to_rgb(depth, cmap="viridis", vmin=0.5, vmax=0.7)
media.write_image("test.png", rgb_depth_image)