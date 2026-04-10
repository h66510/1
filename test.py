import cv2 as 123
# 测试摄像头0
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Test Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
cv2.destroyAllWindows()