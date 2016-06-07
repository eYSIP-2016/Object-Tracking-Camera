############################################
## Import numpy for numerical calculations
import numpy as np

## Import OpenCV for image processing
import cv2

############################################
## Initialize webcam
cap = cv2.VideoCapture(0) ##use 1 in parameter instead of 0 for external
                          ##camera
## You can also use a video by giving url of video in the parameter
## If video is on the same location use only name of the video as
#cap = cv2.VideoCapture('sample.mov')
##If you are using video then comment previous statement

############################################
## param1 and param2 are minimum and maximum range of hsv values for
## green color
param1 = [50,50,50]      ## [H_min,S_min,V_min]
param2 = [90,255,255]    ## [H_max,S_max,V_max]
## You can put range of any color and track that object

############################################
## np.array will change param1 and param2 into numpy array which
## OpenCV can understand
lower = np.array(param1)
upper = np.array(param2)

############################################
## Video Loop

while(1):

    ## Read the frame of video
    ## frame contains frame of the video
    ## ret contains True if frame is successfully read otherwise it
    ## contains False
    ret, frame = cap.read()

    ## If frame is successfully read
    if(ret):

        ## This statement changes color space of frame from BGR to HSV
        # and stores frame into array hsv
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        ## This statement changes all the pixels of the HSV frame into
        # white pixel which lies in the specified range
        ## And changes others into black pixels.
        ## And stores the new frame into mask
        mask  = cv2.inRange(hsv, lower, upper)

        ## This statement removes noise from the Masked Frame (mask)
        mask = cv2.GaussianBlur(mask,(5,5),0)

        ## This statement finds contours (white areas) in mask
        ## And returns all the contours in contours array
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        ## Let area of largest contour is zero
        max_contour_area=0

        ############################################
        ## Colored object tracking

        ## If there is specified colored object in the frame then there
        #will be atleast one contour

        ## If length of contours is atleast 1 then only it will find
        #index of largest contour

        ## And track the object
        if(len(contours) >= 1):
            ## Finding index of largest contour among all the contours
            #for color object tracking
            for i in range(0,len(contours)):
                if(cv2.contourArea(contours[i]) > max_contour_area):
                    max_contour_area = cv2.contourArea(contours[i])
                    max_contour_area_index = i

            ## This statement gives co-ordinates of North-West corner
            #in x and y
            ## And Width and Height in w and h of bounding rectangle
            #of Colored object
            ## If and only if your frame has largest thing of specified
            #color as your object which you want to track
            x,y,w,h=cv2.boundingRect(contours[max_contour_area_index])

            ## This statement will create rectangle around the object
            #which you want to track
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

        ## Showing the video which object tracking
        cv2.imshow('video',frame)

        ## Your video will stop if you press Escape (esc) key
        if cv2.waitKey(60) == 27:  ## 27 - ASCII for escape key
            break

    ## If frame is not successfully read or there is no frame to read
    #(in case of recorded video) stop video
    else:
        break

## Releasing camera
cap.release()

## Destroy all open windows
cv2.destroyAllWindows()
