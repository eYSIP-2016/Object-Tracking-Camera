import numpy as np
import cv2
import get_points

#Minimum no of matches should be there for tracking.
MIN_MATCH_COUNT = 10
FRAME_TIME = 1

#Object of Scale-Invariant Feature Transform Feature detector and descripter.
sift = cv2.SIFT()

#Object of Speeded-Up Robust Features detector and descripter.
surf = cv2.SURF(50)
surf.extended = True

FLANN_INDEX_KDTREE = 0 #for sift, surf etc

#FLANN_INDEX_LSH = 6  #for orb

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) #for sift, surf etc
#index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1) #for orb
search_params = dict(checks = 50)

#Object of Fast Approximate Nearest Neighbor Search Library
flann = cv2.FlannBasedMatcher(index_params, search_params)


cap = cv2.VideoCapture(0)

while(1):
    ret,frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
        #press 'p' to give ROI
        if cv2.waitKey(FRAME_TIME) == ord('p'):
            cv2.destroyWindow('frame')
            points = get_points.run(frame)
            break

roi = frame[points[0][1]:points[0][3],points[0][0]:points[0][2]]
gray_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

h,w = gray_roi.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

#Feature detection and description using SIFT            
kp1, des1 = sift.detectAndCompute(gray_roi,None)

#Feature detection and description using SURF
kp2, des2 = surf.detectAndCompute(gray_roi,None)
print kp1
print kp2

#Combining keypoints and descriptors
kp_roi = np.hstack((kp1,kp2))
des_roi = np.vstack((des1,des2))

#Tracking loop
while(1):
    ret,frame = cap.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Feature detection and description in frame
        kp1, des1 = sift.detectAndCompute(gray_frame,None)
        kp2, des2 = surf.detectAndCompute(gray_frame,None)

        #Combining keypoints and descriptors
        kp_frame = np.hstack((kp1,kp2))
        des_frame = np.vstack((des1,des2))

        #Getting nearest matches using FLANN library
        matches = flann.knnMatch(des_roi,des_frame,k=2)

        good = []
        #Outlier Filtering by nearest neighbour distance ratio
        if len(matches) > 1:
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

        #On getting enough good matches
        if len(good)>=MIN_MATCH_COUNT:
            #Getting matched keypoint's location in ROI
            src_pts = np.float32([ kp_roi[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

            #Getting matched keypoint's location in current frame
            dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            #Getting perspective transform
            dst = cv2.perspectiveTransform(pts,M)

            #Drawing rotating rectangle around object
            cv2.polylines(frame,[np.int32(dst)],True,(255,255,255),3, cv2.CV_AA)
        
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)

        cv2.imshow('tracking',frame)

        if cv2.waitKey(FRAME_TIME) == 27:
            break

cap.release()
cv2.destroyAllWindows()
