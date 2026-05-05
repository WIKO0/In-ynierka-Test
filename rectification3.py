import cv2
import numpy as np

# =========================
# LOAD CALIBRATION
# =========================

K1 = np.load("K1.npy")
d1 = np.load("d1.npy")

K2 = np.load("K2.npy")
d2 = np.load("d2.npy")

K3 = np.load("K3.npy")
d3 = np.load("d3.npy")

# RELACJE (musisz mieć z stereoCalibrate!)
R12 = np.load("R1.npy")
T12 = np.load("T1.npy")

R13 = np.load("R2.npy")
T13 = np.load("T2.npy")

img_size = (640, 480)

# =========================
# RECTIFICATION cam0 ↔ cam1
# =========================

R1_12, R2_12, P1_12, P2_12, Q12, _, _ = cv2.stereoRectify(
    K1, d1,
    K2, d2,
    img_size,
    R12, T12,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x_12, map1y_12 = cv2.initUndistortRectifyMap(
    K1, d1, R1_12, P1_12, img_size, cv2.CV_32FC1
)

map2x_12, map2y_12 = cv2.initUndistortRectifyMap(
    K2, d2, R2_12, P2_12, img_size, cv2.CV_32FC1
)

# =========================
# RECTIFICATION cam0 ↔ cam2
# =========================

R1_13, R2_13, P1_13, P2_13, Q13, _, _ = cv2.stereoRectify(
    K1, d1,
    K3, d3,
    img_size,
    R13, T13,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x_13, map1y_13 = cv2.initUndistortRectifyMap(
    K1, d1, R1_13, P1_13, img_size, cv2.CV_32FC1
)

map3x_13, map3y_13 = cv2.initUndistortRectifyMap(
    K3, d3, R2_13, P2_13, img_size, cv2.CV_32FC1
)

# =========================
# CAMERAS
# =========================

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)

# =========================
# LOOP
# =========================

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    ret3, f3 = cap3.read()

    if not ret1 or not ret2 or not ret3:
        break

    # --- rectify 0 ↔ 1
    r1_12 = cv2.remap(f1, map1x_12, map1y_12, cv2.INTER_LINEAR)
    r2_12 = cv2.remap(f2, map2x_12, map2y_12, cv2.INTER_LINEAR)

    # --- rectify 0 ↔ 2
    r1_13 = cv2.remap(f1, map1x_13, map1y_13, cv2.INTER_LINEAR)
    r3_13 = cv2.remap(f3, map3x_13, map3y_13, cv2.INTER_LINEAR)

    cv2.imshow("cam0-1 rect", r1_12)
    cv2.imshow("cam1 rect", r2_12)

    cv2.imshow("cam0-2 rect", r1_13)
    cv2.imshow("cam2 rect", r3_13)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()