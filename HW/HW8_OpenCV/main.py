import cv2 as cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.CascadeClassifier()
capture_io = cv2.VideoCapture(0)


while True:
    _, img = capture_io.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("img", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
capture_io.release()
cv2.destroyAllWindows()
