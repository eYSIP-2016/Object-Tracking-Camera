# import the necessary packages
import numpy as np
import cv2

# (ix,iy) will be north west corner and
# (jx,jy) will be south east corner of ROI rectangle
ix,iy,jx,jy = -1,-1,-1,-1

# mouse callback function to select ROI in the frame
def select_ROI(event,x,y,flags,param):
    
    global ix,iy,jx,jy

    # If mouse left button is down,
    # set (ix,iy) = current co-ordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
    
    # Else if mouse left button is up,
    # set (jx,jy) = current co-ordinates
    elif event == cv2.EVENT_LBUTTONUP:
        jx,jy = x,y
        # Draw rectangle using (ix,iy) and (jx,jy)
        cv2.rectangle(frame,(ix,iy),(jx,jy),(255,0,0),2)

# Grab the reference to the camera
cap = cv2.VideoCapture(0)

# setup the mouse callback
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
# Binding select_ROI with the frame
cv2.setMouseCallback('frame',select_ROI)

# Decides to pause video or not to take ROI
pause = False

# Decides to track object or not
track = False

# keep looping over the frames
while(1):
    # grab the current frame
    ret ,frame = cap.read()

    # check to see if we have reached the end of the
    # video
    if ret == False:
        break

    # If pause is True, then go into ROI selection mode
    while(pause):
        # Show the current frame only
        # As frame is binded with select_ROI mouse callback function
        # So you can select ROI on this frame by mouse dragging
        cv2.imshow('frame',frame)

        # Press space bar after selecting ROI
        # to process the ROI and to track the object
        if cv2.waitKey(1) & 0xff == 32: #ascii value for spacebar

            # To prevent this loop to start again.
            pause = False

            # setup initial location of window
            r,h,c,w = iy , (jy-iy) , ix , (jx-ix)
            track_window = (c,r,w,h)

            # set up the ROI for tracking
            roi = frame[r:r+h, c:c+w]

            # convert it ROI to the HSV color space
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Masking hsv_roi for good results.
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

            # compute a HSV histogram for the ROI
            roi_hist = cv2.calcHist([hsv_roi],[0,1],mask,[180,256],[0,180,0,256])

            # normalize histogram
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            # Setup the termination criteria,
            # either 10 iteration or move by atleast 1 pt
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

            # Makes track = True to start tracking
            track = True

            # To terminate current loop.
            break

    # After ROI computation start tracking
    if track == True:

        # convert the current frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply backprojection on current frame with respect
        # to roi histogram.
        dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180,0,256],1)

        # apply cam shift to the back projection, convert the
        # points to a bounding box, and then draw them
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.cv.BoxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True, [255,0,0], 2)

    # show the frame 
    cv2.imshow('frame',frame)
    k = cv2.waitKey(20) & 0xff

    # If spacebar is pressed,
    # puase the frame to take ROI
    if k == 32: #ascii value for spacebar
        pause = True

    # If Escape key is pressed,
    # terminate the video
    elif k == 27: #ascii value for escape key
        break

## Releasing camera
cap.release()

## Destroy all open windows
cv2.destroyAllWindows()
