
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import os

from keras.models import load_model
import keras
from PIL import Image




root='D:\\Proje 2'
model = load_model(root+'/Model_200x200x1_Cinsiyet.h5')




face_cascade=cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
Id_Gen = {0: 'Erkek', 1: 'Bayan'}
Gen_Id = dict((g, i) for i, g in Id_Gen.items())

   
sonuc=[]
while True:
  ret , frame=cap.read()
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces=face_cascade.detectMultiScale(gray ,1.3 ,5)

  for(x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    cv2.imwrite('C:\\Users\OmerF\\.spyder-py3\\face.jpg',roi_gray)
    font = cv2.FONT_HERSHEY_DUPLEX
    test_img = Image.open('C:\\Users\OmerF\\.spyder-py3\\face.jpg')
    img = test_img.resize((200,200))
    img = np.array(img) / 255.0
    img = img.reshape(-1,200,200,1)
    age,race,gender = model.predict(img)
    #print("Age:",int(age[0]*100),"Gender:",Id_Gen[gender.argmax()])
    cv2.putText(frame,str(int(age[0]*100))+'  '+str(Id_Gen[gender.argmax()]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(frame,'Cikmak icin "q" basiniz..', (x-50, y-50), font, 1, (255, 255, 255), 2)

  cv2.imshow("frame",frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
   break
cap.release()
cv2.destroyAllWindows()




