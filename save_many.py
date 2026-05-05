import cv2
import torch

cap = cv2.VideoCapture(1)

frames = []

for _ in range(50):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(torch.from_numpy(frame))

cap.release()

video_tensor = torch.stack(frames)  # [T, H, W, C]
torch.save(video_tensor, "video.pt")