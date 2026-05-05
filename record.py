import cv2
import time

cams = [0, 1, 2]

caps = []
writers = []

# otwieranie kamer
for i, cam_id in enumerate(cams):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Kamera {cam_id} nie działa")
        continue

    caps.append(cap)

    # ustawienia wideo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"camera_{cam_id}.mp4",
        fourcc,
        20.0,
        (width, height)
    )

    writers.append(out)

print("Nagrywanie... ESC aby zakończyć")

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()

        if not ret:
            continue

        writers[i].write(frame)
        cv2.imshow(f"Kamera {cams[i]}", frame)

    if cv2.waitKey(1) == 27:
        break

for cap in caps:
    cap.release()

for w in writers:
    w.release()

cv2.destroyAllWindows()
print("Zapisano wideo")