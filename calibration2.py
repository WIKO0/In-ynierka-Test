import cv2
import numpy as np
import glob

CHECKERBOARD = (8, 6)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob("calib/cam0_*.png")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Kamera matrix:\n", K)
print("Dystorsja:\n", dist)

np.save("K1.npy", K)
np.save("d1.npy", dist)