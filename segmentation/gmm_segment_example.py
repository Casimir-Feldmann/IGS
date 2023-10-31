# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


def load_samples(rgb_path, mask_path):

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


colors = ["red", "green"]


def make_ellipses(gmm, ax, colors = ["maroon", "darkgreen"]):
    
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

rgb_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/stop_motion_1/rgb"
mask_path = "/home/casimir/ETH/SemesterProject/IGS/dataset/dataset_processed/stop_motion_1/mask_color"

rgb_dirs = [os.path.join(rgb_path, file) for file in sorted(os.listdir(rgb_path))]
mask_dirs = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]

idx = 0
image_path = rgb_dirs[idx]
mask_path = mask_dirs[idx]

X, y = load_samples(image_path, mask_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {
    cov_type: GaussianMixture(
        n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0
    )
    for cov_type in ["spherical", "diag", "tied", "full"]
}

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(
    bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array(
        [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
    )

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    # make_ellipses(estimator, h)

    label_names = ["foreground", "background"]

    for n, color in enumerate(colors):
        data = X[y == n]
        plt.scatter(
            data[:, 0], data[:, 1], s=0.8, color=color, label=label_names[n]
        )
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

    make_ellipses(estimator, h)
    
    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, "Train accuracy: %.2f" % train_accuracy, transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, "Test accuracy: %.2f" % test_accuracy, transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))


plt.show()