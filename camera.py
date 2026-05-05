# import cv2
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Nie można otworzyć kamery")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Brak obrazu z kamery")
#         break
#
#     cv2.imshow("Kamera", frame)
#
#     # ESC żeby wyjść
#     if cv2.waitKey(1) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()



import cv2

cams = [0, 1, 2]  # zmień jeśli trzeba
caps = [cv2.VideoCapture(i) for i in cams]

while True:
    frames = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow(f"Kamera {cams[i]}", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

for cap in caps:
    cap.release()

cv2.destroyAllWindows()
