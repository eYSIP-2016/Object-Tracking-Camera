#import necessary modules
import numpy as np
import cv2
import math
import functions
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from RPIO import PWM
from multiprocessing import Process, Queue

#Initilaizing camera
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(0.1)

##==============================================================================##
#Initializing Servos
#servo GPIO connections
Roll = 23
Pitch = 24
# Upper limit
RollUL = 230
PitchUL = 230
# Lower Limit
RollLL = 60
PitchLL = 60
#initial Position
initroll = ((RollUL - RollLL) / 2) + RollLL
initpitch = ((PitchUL - PitchLL) / 2) + PitchLL
PWM.setup()
PWM.init_channel(0)
#init servos to center
PWM.add_channel_pulse(0, Roll, 0, initroll)
PWM.add_channel_pulse(0, Pitch, 0, initpitch)

RollCPQ = Queue()	# Servo zero current position, sent by subprocess and read by main process
PitchCPQ = Queue()	# Servo one current position, sent by subprocess and read by main process
RollDPQ = Queue()	# Servo zero desired position, sent by main and read by subprocess
PitchDPQ = Queue()	# Servo one desired position, sent by main and read by subprocess
RollSQ = Queue()	# Servo zero speed, sent by main and read by subprocess
PitchSQ = Queue()	# Servo one speed, sent by main and read by subprocess
def P0():	# Process 0 controlles Pan servo
	speed = .1		# Here we set some defaults:
	RollCP = initroll - 1		# by making the current position and desired position unequal,-
	RollDP = initroll		# 	we can be sure we know where the servo really is. (or will be soon)

	while True:
		time.sleep(speed)
		if RollCPQ.empty():			# Constantly update RollCPQ in case the main process needs-
			RollCPQ.put(RollCP)		# 	to read it
		if not RollDPQ.empty():		# Constantly read read RollDPQ in case the main process-
			RollDP = RollDPQ.get()	#	has updated it
		if not RollSQ.empty():			# Constantly read read RollSQ in case the main process-
			RollS = RollSQ.get()	# 	has updated it, the higher the speed value, the shorter-
			speed = .1 / RollS		# 	the wait between loops will be, so the servo moves faster
		if RollCP < RollDP:					# if RollCPQ less than RollDPQ
			RollCP += 1						# incriment RollCPQ up by one
			RollCPQ.put(RollCP)					# move the servo that little bit
			PWM.clear_channel_gpio(0, Roll)
			PWM.add_channel_pulse(0, Roll, 0, RollCP)
			if not RollCPQ.empty():				# throw away the old RollCPQ value,-
				trash = RollCPQ.get()				# 	it's no longer relevent
		if RollCP > RollDP:					# if RollCPQ greater than ServoPanDP
			RollCP -= 1						# incriment RollCPQ down by one
			RollCPQ.put(RollCP)					# move the servo that little bit
			PWM.clear_channel_gpio(0,Roll)
			PWM.add_channel_pulse(0, Roll, 0, RollCP)
			if not RollCPQ.empty():				# throw away the old ROllPanCPQ value,-
				trash = RollCPQ.get()				# 	it's no longer relevent
		if RollCP == RollDP:	        # if all is good,-
			RollS = 1		        # slow the speed; no need to eat CPU just waiting
			

def P1():	# Process 1 controlles Tilt servo using same logic as above
	speed = .1
	PitchCP = initpitch - 1
	PitchDP = initpitch

	while True:
		time.sleep(speed)
		if PitchCPQ.empty():
			PitchCPQ.put(Pitch)
		if not PitchDPQ.empty():
			PitchDP = PitchDPQ.get()
		if not PitchSQ.empty():
			PitchS = PitchSQ.get()
			speed = .1 / PitchS
		if PitchCP < PitchDP:
			PitchCP += 1
			PitchCPQ.put(PitchCP)
			PWM.clear_channel_gpio(0, Pitch)
			PWM.add_channel_pulse(0, Pitch, 0,PitchCP)
			if not PitchCPQ.empty():
				trash = PitchCPQ.get()
		if PitchCP > PitchDP:
			PitchCP -= 1
			PitchCPQ.put(PitchCP)
			PWM.clear_channel_gpio(0, Pitch)
			PWM.add_channel_pulse(0, Pitch, 0, PitchCP)
			if not PitchCPQ.empty():
				trash = PitchCPQ.get()
		if PitchCP == PitchDP:
			PitchS = 1



Process(target=P0, args=()).start()	# Start the subprocesses
Process(target=P1, args=()).start()	#
time.sleep(1)				# Wait for them to start
##============================================================================##


def CamRight( distance, speed ):		# To move right, we are provided a distance to move and a speed to move.
	global RollCP			# We Global it so  everyone is on the same page about where the servo is...
	if not RollCPQ.empty():		# Read it's current position given by the subprocess(if it's avalible)-
		RollCP = RollCPQ.get()	# 	and set the main process global variable.
	RollDP = RollCP + distance	# The desired position is the current position + the distance to move.
	if RollDP > RollUL:		# But if you are told to move further than the servo is built go...
		RollDP = RollUL		# Only move AS far as the servo is built to go.
	RollDPQ.put(RollDP)			# Send the new desired position to the subprocess
	RollSQ.put(speed)			# Send the new speed to the subprocess
	return;

def CamLeft(distance, speed):			# Same logic as above
	global RollCP
	if not RollCPQ.empty():
		RollCP = RollCPQ.get()
	RollDP = RollCP - distance
	if RollDP < RollLL:
		RollDP = RollLL
	RollDPQ.put(RollDP)
	RollSQ.put(speed)
	return;


def CamUp(distance, speed):			# Same logic as above
	global PitchCP
	if not PitchCPQ.empty():
		PitchCP = PitchCPQ.get()
	PitchDP = PitchCP + distance
	if PitchDP > PitchUL:
		PitchDP = PitchUL
	PitchDPQ.put(PitchDP)
	PitchSQ.put(speed)
	return;


def CamDown(distance, speed):			# Same logic as above
	global PitchCP
	if not PitchCPQ.empty():
		PitchCP = PitchCPQ.get()
	PitchDP = PitchCP - distance
	if PitchDP < PitchLL:
		PitchDP = PitchLL
	PitchDPQ.put(PitchDP)
	PitchSQ.put(speed)
	return;



#============================================================================================================


#defining maximum possible shift in area and center admissible.
MAX_CENTER_SHIFT = 100
MAX_AREA_SHIFT = 5000
HUE_BIN = 180
SAT_BIN = 256

#This threshold will be used to not forget sliding window
THRESHOLD = 3

#Variable used to store standard deviation in histogram distance in previous frame.
prev_standard_deviation = 0

#Variable is used to count number of frames
total_frames = 1

#This kernel will be used in erosion and dilation (opening)
kernel = np.ones((3,3),np.uint8)

#Flag to indicate first frame
first_frame = True

#Termination criteria for CAMShift to stop.
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

#Window to show tracking
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)



#Loop to get first bounding box
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    frame= image.array

    #Show frame
    cv2.imshow('frame',frame)

    #Show some messages
    if first_frame:
        print 'press p to give roi'
        print 'press ESC to exit'
        first_frame = False
        
    k = cv2.waitKey(1)
    #If user presses p, get track window
    if k == ord('p'):
        (c,r,w,h) = functions.get_track_window(frame.copy())
        rawCapture.truncate(0)
        break

    #If user presses escape, exit program
    elif k == 27:
        rawCapture.truncate(0)     
        cv2.destroyAllWindows()
        exit(0)
    rawCapture.truncate(0)
#print w*h
#print (c+w/2,r+h/2)

#Get height, width of the frame.
(height_frame,width_frame,channels) = frame.shape

#Change color space of frame from RGB to HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Use 50 pixels margin from bounding box for tracking
#Make white other area
if r > 50:
    hsv_frame[:r-50,:] = [255,255,255]
if r+h+50 < height_frame:
    hsv_frame[r+h+50:,:] = [255,255,255]
if c > 50:
    hsv_frame[:,:c-50] = [255,255,255]
if c+w+50 < width_frame:
    hsv_frame[:,c+w+50:] = [255,255,255]

#Mask frame for better result
mask_frame = cv2.inRange(hsv_frame, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

#Get Region of interest according to bounding box
hsv_roi = hsv_frame[r:r+h, c:c+w]
mask_roi = mask_frame[r:r+h, c:c+w]

#Get 2D roi histogram and normalize it
hist_roi = cv2.calcHist([hsv_roi],[0,1],mask_roi,[HUE_BIN,SAT_BIN],[0,180,0,256])
cv2.normalize(hist_roi,hist_roi,0,255,cv2.NORM_MINMAX)

#Get 2D frame histogram and normalize it
hist_frame = cv2.calcHist([hsv_frame],[0,1],mask_frame,[HUE_BIN,SAT_BIN],[0,180,0,256])
cv2.normalize(hist_frame,hist_frame,0,255,cv2.NORM_MINMAX)

#Get mean of histogram distance
prev_mean = cv2.compareHist(hist_roi,hist_frame,method=cv2.cv.CV_COMP_BHATTACHARYYA)

#Get ROI back projection on frame
back_projection = cv2.calcBackProject([hsv_frame],[0,1],hist_roi,[0,180,0,256],1)

#Get track window and apply CAMShift
track_window = (c,r,w,h)
retval, track_window = cv2.CamShift(back_projection, track_window, term_crit)
(c,r,w,h) = track_window
Center=[c+w/2,r+h/2]
if Center[0] != 0:		# if the Center of the frame is not zero 

    if Center[0] > 180:	# The camera is moved diffrent distances and speeds depending on how far away-
        CamRight(1,1)
    if Center[0] > 190:	#
        CamRight(2,2)	#
    if Center[0] > 200:	#
        CamRight(6,3)
    if Center[0] > 230:	#
        CamRight(8,3)    #

    if Center[0] < 140:	# and diffrent dirrections depending on what side of center if finds
        CamLeft(1,1)
    if Center[0] < 130:
        CamLeft(2,2)
    if Center[0] < 120:
        CamLeft(6,3)
    if Center[0] < 90:	#
        CamLeft(8,3)  

    if Center[1] > 140:	# and moves diffrent servos depending on what axis we are talking about.
        CamUp(1,1)
    if Center[1] > 150:
        CamUp(2,2)
    if Center[1] > 160:
        CamUp(6,3)
    if Center[1] >190:
        CamUp(8,3)    

    if Center[1] < 100:
        CamDown(1,1)
    if Center[1] < 90:
        CamDown(2,2)
    if Center[1] < 80:
        CamDown(6,3)
    if Center[1] < 50:
        CamDown(8,3)

(c,r,w,h) = (140,100,40,40)
track_window = (140,100,40,40)
prev_area = 40*40
prev_center = (160,120)
#print '******'
#print prev_center
#print prev_area
#print prev_mean
#print '******'

#Flag to get whether roi is lost or not.
isLost = False

#Main loop

for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    frame= image.array
    #Change color space of frame from RGB to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #If roi is not lost, Track ROI
    if not isLost:
        #Increment frame number
        total_frames = total_frames + 1

        #Use 50 pixels margin from bounding box for tracking
        #Make white other area
        if r > 50:
            hsv_frame[:r-50,:] = [255,255,255]
        if r+h+50 < height_frame:
            hsv_frame[r+h+50:,:] = [255,255,255]
        if c > 50:
            hsv_frame[:,:c-50] = [255,255,255]
        if c+w+50 < width_frame:
            hsv_frame[:,c+w+50:] = [255,255,255]

        #Mask frame for better result
        mask_frame = cv2.inRange(hsv_frame, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

        #Get 2D frame histogram and normalize it
        hist_frame = cv2.calcHist([hsv_frame],[0,1],mask_frame,[HUE_BIN,SAT_BIN],[0,180,0,256])
        cv2.normalize(hist_frame,hist_frame,0,255,cv2.NORM_MINMAX)

        #Get histogram similarity between rame and ROI
        histogram_distance = cv2.compareHist(hist_roi,hist_frame,method=cv2.cv.CV_COMP_BHATTACHARYYA)
        current_mean = prev_mean + ((histogram_distance - prev_mean) / total_frames)

        #Get Standard Deviation of Histogram distance
        current_standard_deviation = math.sqrt((((total_frames-2) * (prev_standard_deviation**2)) + ((histogram_distance-prev_mean) * (histogram_distance-current_mean))) / (total_frames-1))

        #Threshold for Histogram Distance
        adaptive_threshold = (current_mean + THRESHOLD * current_standard_deviation) + 0.1

        #print 'histogram_distance = '
        #print histogram_distance
        #print 'adaptive_thes = '
        #print adaptive_threshold

        #ROI lost if 
        if histogram_distance > adaptive_threshold :
            isLost = True
            print 'lost adaptive'

        else:
            #Circulating variables
            prev_mean = current_mean
            prev_standard_deviation = current_standard_deviation

            #Get ROI back projection on frame
            back_projection = cv2.calcBackProject([hsv_frame],[0,1],hist_roi,[0,180,0,256],1)

            #Apply CAMShift
            retval, track_window = cv2.CamShift(back_projection, track_window, term_crit)
            (c,r,w,h) = track_window

            if c<=0 or r<=0 or w<= 0 or h<=0:
                    isLost = True
                    print 'lost due to track_window'

            else:        
                    Center = (c+w/2,r+h/2)

                    cv2.circle(frame,Center,6,(255,0,255),-1)

                    if Center[0] != 0:              # if the Center of the face is not zero 

                            if Center[0] > 180: # The camera is moved diffrent distances and speeds depending on how far away-
                                CamRight(1,1)
                            if Center[0] > 190: #
                                CamRight(2,2)   #
                            if Center[0] > 200: #
                                CamRight(6,3)
                            if Center[0] > 230: #
                                CamRight(8,3)    #

                            if Center[0] < 140: # and diffrent dirrections depending on what side of center if finds
                                CamLeft(1,1)
                            if Center[0] < 130:
                                CamLeft(2,2)
                            if Center[0] < 120:
                                CamLeft(6,3)
                            if Center[0] < 90:  #
                                CamLeft(8,3)  

                            if Center[1] > 140: # and moves diffrent servos depending on what axis we are talking about.
                                CamUp(1,1)
                            if Center[1] > 150:
                                CamUp(2,2)
                            if Center[1] > 160:
                                CamUp(6,3)
                            if Center[1] >190:
                                CamUp(8,3)    

                            if Center[1] < 100:
                                CamDown(1,1)
                            if Center[1] < 90:
                                CamDown(2,2)
                            if Center[1] < 80:
                                CamDown(6,3)
                            if Center[1] < 50:
                                CamDown(8,3)    
                    (c,r,w,h) = (140,100,40,40)
                    track_window = (140,100,40,40)
                    prev_area = 40*40
                    prev_center = (160,120)
            
    #Procedure to re-recognize the object
    else:
        #Get ROI back projection on frame
        back_projection1 = cv2.calcBackProject([hsv_frame],[0,1],hist_roi,[0,180,0,256],1)
        
        #back_projection1 = cv2.GaussianBlur(back_projection,(5,5),0)

        #Create binary image using OTSU algorithm
        ret3,back_projection1 = cv2.threshold(back_projection1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #Apply erosion and dilation
        back_projection1 = cv2.morphologyEx(back_projection1, cv2.MORPH_OPEN, kernel)


        #Get contours
        contours, hierarchy = cv2.findContours(back_projection1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #Get good contours
        good = []
        for i in range(len(contours)):
            #area = cv2.contourArea(contours[i])
            #if area > 0.1 * prev_area:
            good.append(i)

        '''
        #if there is only one contour that is ROI
        if len(good) == 1:
            print 'a'
            c,r,w,h = cv2.boundingRect(contours[good[0]])
            track_window = (c,r,w,h)
            prev_area = w*h
            prev_center = (c+w/2,r+h/2)
            if c > 0 and r > 0 and w > 0 and h > 0:
                isLost = False
                print 'found'
        '''

        #If some good contours found
        if len(good) > 0:
            print 'b'
            #print len(good)
            #print 'some good contours found'

            #List for bounding box
            bounding_box = [[0]*4]*len(contours)

            #List for histogram distance
            hist_dist = [60000]*len(contours)

            #Loop to get histogram distance of each good contour from roi_hist
            for i in good:
                bounding_box[i][0],bounding_box[i][1],bounding_box[i][2],bounding_box[i][3] = cv2.boundingRect(contours[i])
                roi_detected = hsv_frame[bounding_box[i][1]:bounding_box[i][1]+bounding_box[i][3], bounding_box[i][0]:bounding_box[i][0]+bounding_box[i][2]]
                mask_roi_detected = cv2.inRange(roi_detected, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist_roi_detected = cv2.calcHist([roi_detected],[0,1],mask_roi_detected,[HUE_BIN,SAT_BIN],[0,180,0,256])
                cv2.normalize(hist_roi_detected,hist_roi_detected,0,255,cv2.NORM_MINMAX)
                hist_dist[i] = cv2.compareHist(hist_roi,hist_roi_detected,cv2.cv.CV_COMP_BHATTACHARYYA)

            #Get closest histogram distance
            min_hist_dist = min(hist_dist)

            #Get track window if
            if min_hist_dist < adaptive_threshold:
                i = hist_dist.index(min_hist_dist)
                c = bounding_box[i][0]
                r = bounding_box[i][1]
                w = bounding_box[i][2]
                h = bounding_box[i][3]
                track_window = (c,r,w,h)

                Center = (c+w/2,r+h/2) 

                if Center[0] != 0:              # if the Center of the face is not zero 

                    if Center[0] > 180: # The camera is moved diffrent distances and speeds depending on how far away-
                        CamRight(1,1)
                    if Center[0] > 190: #
                        CamRight(2,2)   #
                    if Center[0] > 200: #
                        CamRight(6,3)
                    if Center[0] > 230: #
                        CamRight(8,3)    #

                    if Center[0] < 140: # and diffrent dirrections depending on what side of center if finds
                        CamLeft(1,1)
                    if Center[0] < 130:
                        CamLeft(2,2)
                    if Center[0] < 120:
                        CamLeft(6,3)
                    if Center[0] < 90:  #
                        CamLeft(8,3)  

                    if Center[1] > 140: # and moves diffrent servos depending on what axis we are talking about.
                        CamUp(1,1)
                    if Center[1] > 150:
                        CamUp(2,2)
                    if Center[1] > 160:
                        CamUp(6,3)
                    if Center[1] >190:
                        CamUp(8,3)    

                    if Center[1] < 100:
                        CamDown(1,1)
                    if Center[1] < 90:
                        CamDown(2,2)
                    if Center[1] < 80:
                        CamDown(6,3)
                    if Center[1] < 50:
                        CamDown(8,3)    
                (c,r,w,h) = (140,100,40,40)
                track_window = (140,100,40,40)
                prev_area = 40*40
                prev_center = (160,120)

                
                #If valid bounding box
                if c > 0 and r > 0 and w > 0 and h > 0:
                    isLost = False
                    print 'found'
    print str(Center[0]) + "," + str(Center[1])               
    #Show frame
    cv2.imshow('frame',frame)
    
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xff == 27: #ascii value for escape key
        break

#Release Servos
PWM.clear_channel(0)
PWM.cleanup() 
cv2.destroyAllWindows()
