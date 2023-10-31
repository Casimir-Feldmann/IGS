import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import sys
parent = os.path.dirname(os.getcwd())
import_dir = parent + "/evaluation"
sys.path.append(import_dir)
from evaluation_metrics import calc_IoU_Sets


def generate_labeled_data(rgb_path, mask_path):

    img = cv2.imread(rgb_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    input_mask[input_mask > 0] = 1
    input_mask[input_mask == 0] = 0


    fg = img[input_mask > 0]
    bg = img[input_mask == 0]

    X = np.concatenate((fg, bg), axis=0)
    y = np.concatenate((np.ones(fg.shape[0]), np.zeros(bg.shape[0])), axis=0)

    return X, y

def load_data(rgb_path):

    img = cv2.imread(rgb_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    X = img.reshape(-1, 3)

    return X

def train_gm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    n_classes = len(np.unique(y_train))

    gm = GaussianMixture(n_components=n_classes, covariance_type="spherical", max_iter=20, random_state=0)

    gm.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])

    gm.fit(X_train)
        
    y_train_pred = gm.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel())

    y_test_pred = gm.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel())

    print("Train Acc:", train_accuracy, "Test Acc:", test_accuracy)

    return gm


rgb_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/stop_motion_1/rgb"
mask_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/stop_motion_1/mask_color"

rgb_dirs = [os.path.join(rgb_path, file) for file in sorted(os.listdir(rgb_path))]
mask_dirs = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]

idx = 0
image_path = rgb_dirs[idx]
mask_path = mask_dirs[idx]

X, y = generate_labeled_data(image_path, mask_path)

gm = train_gm(X, y)

IoU_list = []

for image_path, mask_path in zip(rgb_dirs, mask_dirs):

    X = load_data(image_path)

    y_pred = gm.predict(X)

    img = cv2.imread(image_path)
    mask_pred = y_pred.reshape((img.shape[0], img.shape[1], 1)).squeeze()

    input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    input_mask[input_mask > 0] = 1
    input_mask[input_mask == 0] = 0

    IoU = calc_IoU_Sets(truth=input_mask, pred=mask_pred)
    IoU_list.append(IoU)


print("IoU on all", sum(IoU_list)/len(IoU_list))