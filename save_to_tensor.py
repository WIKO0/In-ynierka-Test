import cv2
import torch

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

# BGR → RGB
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# numpy → tensor
tensor = torch.from_numpy(frame)

# zapis
torch.save(tensor, "frame.pt")

print("Zapisano")