import cv2
import numpy as np
import glob

CHECKERBOARD = (8,6)

# --- przygotowanie punktów 3D (wzorzec)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []
imgpoints_l = []
imgpoints_r = []

images_l = sorted(glob.glob("calib/cam0_*.png"))
images_r = sorted(glob.glob("calib/cam2_*.png"))

for imgL, imgR in zip(images_l, images_r):

    imgL = cv2.imread(imgL)
    imgR = cv2.imread(imgR)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_l.append(cornersL)
        imgpoints_r.append(cornersR)

        cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)

        cv2.imshow("L", imgL)
        cv2.imshow("R", imgR)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# --- TU wstaw swoje K i distortion z kalibracji
K1 = np.load("K1.npy")
d1 = np.load("d1.npy")
K3 = np.load("K3.npy")
d3 = np.load("d3.npy")

# stereo calibration
ret, K1, d1, K3, d3, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    K1, d1,
    K3, d3,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
)

print("Rotation:\n", R)
print("Translation:\n", T)

np.save("R2.npy", R)
np.save("T2.npy", T)