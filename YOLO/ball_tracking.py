#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:31:39 2018

@author: seanlin
"""

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math
import os
import pandas as pd


FILE_INPUT = './../Hoop/'

FILE_OUTPUT = './../Hoop/'
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
Lower = (4, 80, 145)
Upper = (14, 145, 240)



# allow the camera or video file to warm up
time.sleep(2.0)



display = True
pause = False

#videos = os.listdir(FILE_INPUT)
#videos.remove('.DS_Store')
videos = [#'IMG_5138_demo.avi', 
          'IMG_5139_demo.avi']#, 'IMG_5144_demo.avi', 'IMG_5151_demo.avi', 
          #'IMG_5153_demo.avi', 
          #'IMG_5164_demo.avi', 
          #'IMG_5165_demo.avi']

# keep looping
for video in os.listdir(FILE_INPUT) : 
    if video in videos:
        pts.clear()
        vs = cv2.VideoCapture((FILE_INPUT + video))
        name = video.split('.')[0]
        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        output = os.path.join(FILE_OUTPUT, name + '+bt.avi')
        out = cv2.VideoWriter(output, fourcc, round(vs.get(cv2.CAP_PROP_FPS)), (1920, 1080))
        # import CSV
        CSV_dir = './../Hoop/video2frame/' + name.rsplit('_', 1)[0] + '/'
        hoop = pd.read_csv((CSV_dir + name + '.csv'), header = 0)
        y_top = 1200.0
        x_max = -1.0
        x_min = 1920.0
        total = 0
        for i in range(len(hoop)):
            if(hoop["label"][i] != 'NAN'):
                if float(hoop["y1"][i]) < y_top:
                    y_top = float(hoop["y1"][i])
                if float(hoop["x1"][i]) < x_min:
                    x_min = float(hoop["x1"][i])
                if float(hoop["x2"][i]) > x_max:
                    x_max = float(hoop["x2"][i])

        x_avg = (x_min + x_max) / 2

        print(video)
#         print(x_avg)
#         print(y_top)
        
        display = True
        pause = False
        max_radius = 0
        min_y = 1200
        while display == True :
            while pause == False : 
                # grab the current frame
                frame = vs.read()

                # handle the frame from VideoCapture or VideoStream
                #frame = frame[1] if args.get("video", False) else frame
                frame = frame[1]

                # if we are viewing a video and we did not grab a frame,s
                # then we have reached the end of the video
                if frame is None:
                    display = False
                    break

                # resize the frame, blur it, and convert it to the HSV
                # color space
                #frame = imutils.resize(frame, width=600)
                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #        print(hsv.shape)

                # construct a mask for the color "green", then perform
                # a series of dilations and erosions to remove any small
                # blobs left in the mask
                t = 315
                mask = cv2.inRange(hsv[:300,t:], Lower, Upper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # find contours in the mask and initialize the current
                # (x, y) center of the ball
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                center = None
                modify_center = np.array([0, 0])

                # only proceed if at least one contour was found
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # only proceed if the radius meets a minimum size
                    if radius > 10 :
                        #draw the circle and centroid on the frame,
                        #update the list of tracked points
                        cv2.circle(frame, (int(x+t), int(y)), int(radius),
                            (0, 255, 255), 2)
                        modify_center = (np.asarray(center)+ [t, 0])
                        cv2.circle(frame, tuple(modify_center), 5, (0, 0, 255), -1)

                        if radius > max_radius : 
                            max_radius = radius

                # update the points queue
                pts.appendleft(tuple(modify_center))
#                 print((tmp, tmp.shape[0]))
                if modify_center.shape:
                    if modify_center[1] < min_y and modify_center[0] < 500:
                        min_y = modify_center[1]
#                         print(min_y)
                
                # loop over the set of tracked points
                backward = [0,0]
                for i in range(1, len(pts)):
                    #if either of the tracked points are None, ignore them
                    if pts[i - 1] is None or pts[i] is None:
                        continue     

                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

                    if isinstance(pts[i], tuple) and isinstance(pts[i-1], tuple):
                        if (pts[i - 1][0]-pts[i][0])**2+(pts[i - 1][1]-pts[i][1])**2 < 10000:
                            forward = (np.asarray(pts[i - 1]))
                            backward = (np.asarray(pts[i]))
#                             print("backward = ",backward)
                            cv2.line(frame,  pts[i-1], pts[i], (0, 0, 255), math.floor(thickness/2))

                out.write(frame)
                # show the frame to our screen
                cv2.imshow("Frame", frame)

        #        print(frame.shape)

                if ((modify_center[0]-x_avg)**2 + (modify_center[1]-y_top)**2) < (max_radius**2):
                    #=======calculate angle==========
                    print("center = ", modify_center)
                    tmp_point = (0, 0)
                    for i in range(1, len(pts)-1):
                        if isinstance(pts[i], tuple):
#                             print("pts = ", pts[i])
                            if math.fabs(pts[i][0] - modify_center[0]) > 10 and math.fabs(pts[i][1] - modify_center[1]) > 10:
                                tmp_point = pts[i]
                                print("tmp_point = ", tmp_point)
                                print("center = ", modify_center)
                                break
            
                    slope = (tmp_point[1] - modify_center[1]) / float(tmp_point[0] - modify_center[0])
                    top = (math.ceil(modify_center[0] - modify_center[1] / slope), 0)
                    theta = math.fabs(round(math.degrees(math.atan(slope)),2))
                    print(theta)
                    cv2.line(frame, tuple(modify_center), tuple(top), (43, 255, 255), thickness*3)
                    cv2.line(frame, tuple(modify_center), (500, modify_center[1]), (43, 255, 255), thickness*3)
                    if theta < 30:
                        cv2.putText(frame, str(theta), tuple(modify_center + [65, -10]), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(theta), tuple(modify_center + [40, -10]), 
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    print('pause')
                    pause = True
                    out.write(frame)
                    cv2.imshow("Frame", frame)
                    out.release()
                    break
                    #================================
                    
                
                key = cv2.waitKey(1) & 0xFF

                    # if the 'q' key is pressed, stop the loop
                if key == ord("q"):
                    display = False
                    break

                if key == ord("p"):
                    pause = True
                    break


            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                pause = False

            if key == ord("q"):
                display = False

    
                

 
## if we are not using a video file, stop the camera video stream
#if not args.get("video", False):
#    vs.stop()
# 
## otherwise, release the camera
#else:
#	vs.release()
            
#vs.stop()
 
    

# close all windows
cv2.destroyAllWindows()



        
        











