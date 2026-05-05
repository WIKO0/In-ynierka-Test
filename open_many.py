import torch
import cv2

video = torch.load("video.pt")

paused = False

for frame in video:
    # jeśli pauza → czekaj aż użytkownik kliknie coś
    while paused:
        key = cv2.waitKey(0) & 0xFF

        if key == ord(' '):  # spacja = resume
            paused = False
        elif key == 27:  # ESC = wyjście
            cv2.destroyAllWindows()
            exit()

    frame = frame.numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("video.pt", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord(' '):  # spacja = pauza
        paused = True

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()