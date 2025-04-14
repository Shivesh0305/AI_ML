import cv2, time, os, HandTrackingModule as htm

# Turns on Camera
cap= cv2.VideoCapture(0)

# Setting the size of the camera
wCam, hCam=640,480
cap.set(3, wCam)
cap.set(4, hCam)

# Putting training images in a list
folderPath="FingerImages"
myList=os.listdir(folderPath)
overLayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

pTime=0

# detects hand movements
detector=htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20]
while True:
    # Reads the image from camera
    success, img=cap.read()

    # finds the hand in camera
    img=detector.findHands(img)

    # find the position of hand
    lmList=detector.findPosition(img)
    
    # checks if the finger tip number is below the other point on the same finger (the numbers are available on mediapie web page)
    if len(lmList)!=0:
        fingers=[]
        # for thumb check if left or right of the given point
        if lmList[tipIds[0]][1]<lmList[tipIds[0]-11][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers=fingers.count(1)

    # h,w,c takes the size of the training image and displays it at the location as specified ([0:h,0:w] in this case)
        h,w,c=overLayList[totalFingers].shape
        img[0:h, 0:w]=overLayList[totalFingers]


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    # prints fps:-
    # location (0,460) 
    # font FONT_HERSHEY_COMPLEX
    # scale 0.7
    # colour (0,0,255)
    # thickness 2
    cv2.putText(img, f'FPS:{int(fps)}',(0,460),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    cv2.imshow("Image",img)

    # Provides delay of 1 millisecond
    cv2.waitKey(1)

