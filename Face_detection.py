#Just added some features to this existing code 
#after modification it can also recognise the smile and eyes
#Which means now it is a proper human face recognisor !!


import cv2
from random import randrange 


trained_face_data= cv2.CascadeClassifier('Face_data.xml')
trained_eye_data = cv2.CascadeClassifier('Eye_data.xml')
trained_smile_data = cv2.CascadeClassifier('Smile_data.xml')



webcam = cv2.VideoCapture(0)


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2



while 1:
      s_F_R,frame=webcam.read()
      grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      cordinates=trained_face_data.detectMultiScale(grayscaled_img)
      
      for (x,y,w,h) in cordinates:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,254),3)
          cv2.putText(frame, 'Human', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.75, (0, 255, 0), 2)
          roi_gray = grayscaled_img[y:y+h, x:x+w]
          roi_color =         frame[y:y+h, x:x+w]

          smile = trained_smile_data.detectMultiScale(
              roi_gray,
              scaleFactor=1.16,
              minNeighbors=35,
              minSize=(25, 25),
              flags=cv2.CASCADE_SCALE_IMAGE
          )
          for (sx, sy, sw, sh) in smile:
              cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

          eyes = trained_eye_data.detectMultiScale(roi_gray)
          for (ex, ey, ew, eh) in eyes:
              cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

             

      cv2.imshow('Face detector',frame)
      key=cv2.waitKey(1)

      if key==81 or key==113:
        break;


