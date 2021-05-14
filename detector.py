import numpy as np
import cv2 

faceCascade = cv2.CascadeClassifier('Cascade/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascade/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascade/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

face_id = input('\n nhập id :')
print("\n Nhìn vào camera và chờ xử lý...")

count = 0



while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image',img)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale (
            roi_gray,
            scaleFactor = 1.5,
            minNeighbors = 5,
            minSize = (5, 5),
            )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25,25),
            )

        for(xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx,yy), (xx+ww, yy + hh), (0, 255, 0), 2)

    cv2.imshow('video',img)
    cv2.imshow('gray',gray)

    k = cv2.waitKey(60) & 0xff
    if k == 27: # bam 'ESC' de thoat
        break
    elif count >= 60: 
        break

print("\n Thoát chương trình")
cap.release()
cv2.destroyAllWindows()