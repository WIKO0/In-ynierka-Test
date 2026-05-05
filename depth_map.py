import cv2
import numpy as np

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# lepszy algorytm niż StereoBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,   # musi być wielokrotnością 16
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if not ret1 or not ret2:
        break

    # grayscale (wymagane)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(g1, g2).astype(np.float32) / 16.0

    # normalizacja do wizualizacji
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # kolor
    depth_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    cv2.imshow("cam0", f1)
    cv2.imshow("cam1", f2)
    cv2.imshow("depth map", depth_color)

    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()