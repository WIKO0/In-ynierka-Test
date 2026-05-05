import cv2
import torch
import numpy as np

cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

frames0, frames1, frames2 = [], [], []

print("Naciśnij 'q' aby zakończyć nagrywanie")

while True:
    ret0, f0 = cap0.read()
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if not ret0 or not ret1 or not ret2:
        break

    cv2.imshow("cam0", f0)
    cv2.imshow("cam1", f1)
    cv2.imshow("cam2", f2)

    frames0.append(f0.copy())
    frames1.append(f1.copy())
    frames2.append(f2.copy())

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()

# =========================
# KONWERSJA DO TENSORA
# =========================

v0 = np.stack(frames0, axis=0)  # (T,H,W,3)
v1 = np.stack(frames1, axis=0)
v2 = np.stack(frames2, axis=0)

video_3cam = np.stack([v0, v1, v2], axis=1)  # (T,3,H,W,3)

tensor_3cam = torch.from_numpy(video_3cam).float() / 255.0

print("Shape:", tensor_3cam.shape)
print("dtype:", tensor_3cam.dtype)

torch.save(tensor_3cam, "video_3cam.pt")

print("Zapisano: video_3cam.pt")