import cv2

kamera = cv2.VideoCapture(0)

while True:

    ret, frame = kamera.read()

    if not ret:
        break

    cv2.imshow("Tes Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()