#importing useful modules
import numpy as np
import cv2
import sys

#Global variables for track_window
ix,iy,jx,jy = -1,-1,-1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,jx,jy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        jx,jy = x,y
        cv2.rectangle(frame,(ix,iy),(x,y),(255,0,0),2)

#Initializing camera
cap = cv2.VideoCapture(0)

#Creating object of KNearest class which will be used in feature matching
knn = cv2.KNearest()

#Creating object of ORB class which will be used to extract oriented
#Features from Accelerated Segment Test and Binary Robust Independent
#Elementary Features in the object and frame
orb = cv2.ORB()

#Creating a window
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)

#Regitering frame to mouse callback function
cv2.setMouseCallback('frame',draw_circle)

#Flag to pause and play the video
pause = False

#Flag to start tracking
track = False

#Variables for training
trainResponce = []
train = []

first_time = True
#Main loop
while(1):
    #Reading frame
    ret ,frame = cap.read()
    #on successful read
    if ret == True:
        #On first frame
        if first_time:
            print 'press space to give ROI'
            print 'After giving ROI again press spacebar'
            first_time = False
        #If space is pressed
        while(pause):
            #Show frame
            cv2.imshow('frame',frame)
            #If space is again pressed
            if cv2.waitKey(1) & 0xff == 32: #ascii of spacebar
                pause = False

                # setup initial location of window
                r,h,c,w = iy , (jy-iy) , ix , (jx-ix)
                #c,r,w,h = ix,iy,(jx-ix),(jy-iy)
                #track_window = (c,r,w,h)
                
                #Creating gray scale image
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #Extracting features and descripters from the image
                kpFrame, desFrame = orb.detectAndCompute(frame_gray,None)
                #Positive means inside ROI
                #Negative means outside ROI
                kpPositive = []
                kpNegative = []
                desPositive = []
                desNegative = []
                #Separating Features and descripters of inside and outside of frame.
                for i in range (len(kpFrame)):
                    if kpFrame[i].pt[0] > c and kpFrame[i].pt[0] < c+w and kpFrame[i].pt[1] > r and kpFrame[i].pt[1] < r+h:
                        kpPositive.append(kpFrame[i])
                        desPositive.append(desFrame[i])
                    else:
                        kpNegative.append(kpFrame[i])
                        desNegative.append(desFrame[i])
                #Learning descripters
                train = train + desPositive + desNegative
                #Giving value 1 for positive and value 0 for negative descripters
                trainResponceTemp = [1 for i in (desPositive)]
                trainResponceTemp = trainResponceTemp + [0 for i in (desNegative)]
                trainResponce = trainResponce + trainResponceTemp
                track = True
                break
        #Start tracking
        if track == True:
            #Read next frame
            ret ,frame = cap.read()
            #Gray scale image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #getting features and descripters in this frame as test data
            kpTest, test = orb.detectAndCompute(frame_gray,None)
            
            train = np.asarray(train).astype(np.float32)
            trainResponce = np.asarray(trainResponce).astype(np.float32)
            #Training KNearest object
            knn.train(train,trainResponce)
            test = np.asarray(test).astype(np.float32)
            #Finding 5 nearest neighboursof test data in train data
            ret,result,neighbours,dist = knn.find_nearest(test,k=5)

            kp = []
            nearest_dist = np.array(dist[:,0])
            
            #Outlier Filtering by nearest neighbour distance ratio
            for i in range (len(nearest_dist)-1):
                if nearest_dist[i] < 0.7*nearest_dist[i+1]:
                    result[i+1] = 0.

            #Discarding all matches whose first neighbour is keypoint of background
            for i in range (len(neighbours)):
                if neighbours[i][0] == 0.:
                    result[i] = 0.

            #Saving good keypoints
            for i in range (len(kpTest)):
                if result[i] == 1.:
                    kp.append(kpTest[i])

            #Drawing good keypoints
            frame = cv2.drawKeypoints(frame,kp,color=(0,255,0), flags=0)

        #Showing frame
        cv2.imshow('frame',frame)
        k = cv2.waitKey(60) & 0xff
        if k == 32:
            pause = True
        elif k == 27:
            break
    else:
        break
        
cv2.destroyAllWindows()
cap.release()
