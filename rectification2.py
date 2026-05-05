import cv2
import numpy as np

# --- load calibration ---
K1 = np.load("K1.npy")
d1 = np.load("d1.npy")
K2 = np.load("K2.npy")
d2 = np.load("d2.npy")
R = np.load("R.npy")
T = np.load("T.npy")


img_size = (640, 480)

# --- rectification setup (MUSI BYĆ PRZED PĘTLĄ) ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, d1,
    K2, d2,
    img_size,
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, d1, R1, P1, img_size, cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, d2, R2, P2, img_size, cv2.CV_32FC1
)

# --- cameras ---
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # --- rectification ---
    r1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

    cv2.imshow("cam1 rect", r1)
    cv2.imshow("cam2 rect", r2)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()