import cv2
import get_points
import TLD

cap = cv2.VideoCapture(0)

while(1):
    ret,frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('p'):
            cv2.destroyWindow('frame')
            bounding_box = get_points.run(frame)
            break
        
multiple = 0.5
print bounding_box
bounding_box[0] = bounding_box[0]*multiple
bounding_box[1] = bounding_box[1]*multiple
bounding_box[2] = bounding_box[2]*multiple
bounding_box[3] = bounding_box[3]*multiple

while(1):
    ret,frame = cap.read()
    if not ret:
        break
    last_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    last_gray = cv2.resize(last_gray,(320,240),multiple,multiple)

    tld = TLD.TLD()
    
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
