import cv2
import numpy as np

K1 = np.load("K1.npy")
d1 = np.load("d1.npy")
K2 = np.load("K2.npy")
d2 = np.load("d2.npy")

R = np.load("R.npy")
T = np.load("T.npy")

img_size = (640, 480)  # dostosuj do swoich kamer

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, d1,
    K2, d2,
    img_size,
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, img_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, img_size, cv2.CV_32FC1)