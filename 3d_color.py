import cv2

capL = cv2.VideoCapture(1, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(2, cv2.CAP_DSHOW)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while True:
    retL, left = capL.read()
    retR, right = capR.read()

    if not retL or not retR:
        break

    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR)

    # normalizacja do 0–255
    disp = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp = disp.astype('uint8')

    # KOLORYZACJA
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)

    cv2.imshow("left", left)
    cv2.imshow("right", right)
    cv2.imshow("depth color", disp_color)

    # debug
    cv2.imshow("L", grayL)
    cv2.imshow("R", grayR)

    if cv2.waitKey(1) == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()