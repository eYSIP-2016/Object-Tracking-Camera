# Import the required modules
import cv2
import math

def distance(current_center,prev_center):
    return math.sqrt((current_center[0]-prev_center[0])**2 + (current_center[1]-prev_center[1])**2)


def get_track_window(im):
    cv2.imshow('frame', im)

    # List containing top-left and bottom-right to crop the image.
    pts_1 = []
    pts_2 = []

    rects = []
    get_track_window.mouse_down = False

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
	    if len(pts_2) == 1: #pts_2 contains one tuple.
		print "WARN: Cannot select another object."
		print "Delete the previously selected object using key `d` to mark a new location."
		return
            get_track_window.mouse_down = True
            pts_1.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and get_track_window.mouse_down == True:
            get_track_window.mouse_down = False
            pts_2.append((x, y))
            print "Object selected at [{}, {}]".format(pts_1[-1], pts_2[-1])
        elif event == cv2.EVENT_MOUSEMOVE and get_track_window.mouse_down == True:
            im_draw = im.copy()
            cv2.rectangle(im_draw, pts_1[-1], (x, y), (255,255,255), 3)
            cv2.imshow('frame', im_draw)

    print "Press and release mouse around the object to be tracked."
    cv2.setMouseCallback('frame', callback)

    print "Press key `p` to continue with the selected points."
    print "Press key `d` to discard the last object selected."
    print "Press key `q` to quit the program."
    while True:
        # Draw the rectangular boxes on the image
        for pt1, pt2 in zip(pts_1, pts_2):
            rects.append([pt1[0],pt2[0], pt1[1], pt2[1]])
            cv2.rectangle(im, pt1, pt2, (255, 255, 255), 3)
        cv2.imshow('frame', im)
        key = cv2.waitKey(30)
        if key == ord('p'):
            # Press key `s` to return the selected points
            cv2.destroyAllWindows()
            point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
            corrected_point=check_point(point)
            return corrected_point
        elif key == ord('q'):
            # Press key `q` to quit the program
            print "Quitting without saving."
            exit()
        elif key == ord('d'):
            # Press ket `d` to delete the last rectangular region
            if get_track_window.mouse_down == False and pts_1:
                print "Object deleted at  [{}, {}]".format(pts_1[-1], pts_2[-1])
                pts_1.pop()
                pts_2.pop()
                im_disp = im.copy()
            else:
                print "No object to delete."
    cv2.destroyAllWindows()
    point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
    return check_point(point) 

def check_point(points):
    for point in points:
        #to find min and max x coordinates
        if point[0]<point[2]:
            minx=point[0]
            maxx=point[2]
        else:
            minx=point[2]
            maxx=point[0]
        #to find min and max y coordinates
        if point[1]<point[3]:
            miny=point[1]
            maxy=point[3]
        else:
            miny=point[3]
            maxy=point[1]

    return (minx,miny,maxx-minx,maxy-miny)
