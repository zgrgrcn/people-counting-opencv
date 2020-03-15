import datetime

import cv2
import dlib
import imutils
import time
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

DEBUG = True
#save videos
SAVEOUTPUT = True
CAPTUREFROMWEBCAM = True

input='videos/ex2.mp4'
minConfidence=0.4
skip_frames=30

print("Model Loading...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

print("opening video file...")
if CAPTUREFROMWEBCAM is False:
    vs = cv2.VideoCapture(input)
else:
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
W = 640
H = 360
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
oldtotalDown = 0
oldtotalUp = 0
directionTextArray = []
writer = None

fps = FPS().start()
print('Start Time: {:}'.format(datetime.datetime.now()))

while True:
    frame = vs.read()
    if CAPTUREFROMWEBCAM is False:
        frame = frame[1]
    if frame is None:
        break

    frame = imutils.resize(frame, width=W, height=H)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if SAVEOUTPUT is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('output/output{:}.avi'.format(datetime.datetime.now()), fourcc, 30,(W, H), True)

    rects = []
    #Her skip_frames bir detection yapılıyor çünkü bu maliyetli bir işlem
    if totalFrames % skip_frames == 0:
        trackers = []

        #127.5 => Renklerin ortalaması
        #0.007843 => 1/127.5
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2] #bulunan doğru olma olasığı

            if confidence > minConfidence:
                idx = int(detections[0, 0, i, 1])#Bulunan nesnenin türü

                if idx != 15:#Person
                    continue

                #Bulunan nesnenin konumu, kutu şeklinde değerleri
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    #Tracking
    else:

        for tracker in trackers:
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))

    #Tracking bitti
    if DEBUG == True:
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 255), 2)
    objects = ct.update(rects)

    #objectID => key
    #centroid => value
    for (objectID, centroid) in objects.items():

        directionText = ''
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids] # Dikeye bakıyor, cismin konumların
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if direction < -10 and centroid[1] < H // 2: #-10 is threshold
                    totalUp += 1
                    directionText = 'Up'
                    directionTextArray.append('Up')
                    to.counted = True

                elif direction > 10 and centroid[1] > H // 2:#10 is threshold
                    totalDown += 1
                    directionText = 'Down'
                    directionTextArray.append('Down')
                    to.counted = True

        trackableObjects[objectID] = to
        if DEBUG == True:
            if len(directionTextArray) > objectID:
                cv2.putText(frame, directionTextArray[objectID], (centroid[0] - 10, centroid[1] - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),2)

    if DEBUG == True:
        info = [
            ("Down", totalDown),
            ("Up", totalUp),
    ]
    if DEBUG == True:
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (int(W/20), H - ((i * (H-40)) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Frame", frame)

    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    totalFrames += 1
    if oldtotalDown != totalDown:
        print("Total Down:{:.2f} Time: {:}".format(totalDown,datetime.datetime.now()))
        oldtotalDown = totalDown
    if oldtotalUp != totalUp:
        print("Total Up:{:.2f} Time: {:}".format(totalUp,datetime.datetime.now()))
        oldtotalUp = totalUp

    fps.update()

fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))
print('End Time: {:}'.format(datetime.datetime.now()))


if writer is not None:
    writer.release()
if CAPTUREFROMWEBCAM:
    vs.stop()
else:
    vs.release()
