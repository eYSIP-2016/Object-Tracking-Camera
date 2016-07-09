#Importing necessary modules
import numpy as np
import cv2
import functions
import util
import get_points

#miliseconds for cv2.waiKey(_)
FRAME_TIME = 1

#Object of Consensus based matching and Tracking
CMT = functions.CMT()

cap = cv2.VideoCapture(0)
first_frame = True
print cap.isOpened()

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame',frame)
        if first_frame:
            print 'press p to give roi'
            print 'press ESC to exit'
            first_frame = False
        k = cv2.waitKey(FRAME_TIME) 
        if k == ord('p'):
            tl,br = get_points.run(frame)
            break
        elif k == 27:
            cap.release()
            cv2.destroyWindow('frame')
            exit(0)

frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

CMT.initialise(frame_gray, tl, br)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    CMT.process_frame(frame_gray)

    # Draw updated estimate
    if CMT.has_result:

            #Drawing center of Object
            center = (((CMT.tl[0]+CMT.br[0])/2),((CMT.tl[1]+CMT.br[1])/2))
            (xcenter,ycenter) = center
            cv2.line(frame, (xcenter,ycenter-15), (xcenter,ycenter+15), (0,0,255), 3)
            cv2.line(frame, (xcenter-15,ycenter), (xcenter+15,ycenter), (0,0,255), 3)
            cv2.circle(frame,center, 12, (0,0,255), 3)
            
            #Drawing rectangle around object
            cv2.line(frame, CMT.tl, CMT.tr, (255, 0, 0), 4)
            cv2.line(frame, CMT.tr, CMT.br, (255, 0, 0), 4)
            cv2.line(frame, CMT.br, CMT.bl, (255, 0, 0), 4)
            cv2.line(frame, CMT.bl, CMT.tl, (255, 0, 0), 4)

    #Drawing moving keypoints
    util.draw_keypoints(CMT.tracked_keypoints, frame, (255, 255, 255))
    # this is from simplescale
    util.draw_keypoints(CMT.votes[:, :2], frame)  # blue
    util.draw_keypoints(CMT.outliers[:, :2], frame, (0, 0, 255))

    cv2.imshow('frame',frame)
    if  cv2.waitKey(FRAME_TIME) == 27:
        break

cap.release()
cv2.destroyAllWindows()
