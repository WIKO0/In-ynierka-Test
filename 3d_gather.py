import cv2

cams = [0, 1, 2]
caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in cams]

while True:
    frames = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        frames.append(frame)
        cv2.imshow(f"cam {cams[i]}", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

for cap in caps:
    cap.release()

cv2.destroyAllWindows()