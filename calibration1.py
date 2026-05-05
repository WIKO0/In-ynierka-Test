import cv2
import os

cams = [0, 1, 2]
caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in cams]

os.makedirs("calib", exist_ok=True)

count = 0

while True:
    frames = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow(f"cam {i}", frame)
        frames.append(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # SAVE
        for i, frame in enumerate(frames):
            cv2.imwrite(f"calib/cam{i}_{count}.png", frame)
        print("zapisano:", count)
        count += 1

    if key == 27:  # ESC
        break

for cap in caps:
    cap.release()

cv2.destroyAllWindows()