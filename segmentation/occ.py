import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import matplotlib.colors as colors

import numpy as np
import cv2
import os
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

import sys
parent = os.path.dirname(os.getcwd())
import_dir = parent + "/evaluation"
sys.path.append(import_dir)
from evaluation_metrics import calc_IoU_Sets

def generate_dataset(rgb_path, mask_path):

    img = cv2.imread(rgb_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    input_mask[input_mask > 0] = 1
    input_mask[input_mask == 0] = 0

    inliers = img[input_mask > 0]

    rest = img[input_mask <= 0]

    return inliers, rest

def load_data(rgb_path):

    img = cv2.imread(rgb_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    X = img.reshape(-1, 3)

    return X

def inference_debug(model, scaler, rgb_dirs, save_path):

    for idx, image_path in tqdm(enumerate(rgb_dirs)):

        X = load_data(image_path)

        X_norm  = scaler.transform(X)

        y_pred = model.predict(X_norm)

        img = cv2.imread(image_path)
        mask_pred = y_pred.reshape((img.shape[0], img.shape[1], 1)).squeeze()

        plt.imshow(img)
        plt.imshow(mask_pred, alpha=0.6)
        plt.show()
        # plt.savefig(f"{save_path}/mask_{idx}.png")
        # plt.close()


def pipeline():
    
    rgb_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/2023-10-27-13-20-44/rgb"
    mask_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/2023-10-27-13-20-44/estimated_masks"
    save_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/2023-10-27-13-20-44/gm_mask_debug"
    
    os.makedirs(save_path, exist_ok=True)

    rgb_dirs = [os.path.join(rgb_path, file) for file in sorted(os.listdir(rgb_path))]
    mask_dirs = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]

    idx = 333
    image_path = rgb_dirs[idx]
    mask_path = mask_dirs[idx]

    outliers_fraction=0.5
    n_components = 100

    print("Getting samples")
    X_fg, X_rest = generate_dataset(image_path, mask_path)

    X = np.concatenate((X_fg, X_rest))

    scaler = StandardScaler()
    scaler.fit(X)

    X_fg_norm = scaler.transform(X_fg)


    
    print("Creating OCC:")
    # clf = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1, verbose=True)
    # clf = make_pipeline(
    #         Nystroem(gamma=0.1, random_state=42, n_components=100),
    #         SGDOneClassSVM(
    #             nu=outliers_fraction,
    #             shuffle=True,
    #             fit_intercept=True,
    #             random_state=42,
    #             tol=1e-6,
    #             verbose=True
    #         )
    # )
    clf = EllipticEnvelope(random_state=42, contamination=outliers_fraction)

    print("Now fitting on initial data:")
    clf.fit(X_fg_norm)

    # y_pred = clf.predict(X)



    # data_plot(X, y_pred, gm)

    inference_debug(clf, scaler, rgb_dirs, save_path)



def main():
    pipeline()



if __name__ == "__main__":
    main()

