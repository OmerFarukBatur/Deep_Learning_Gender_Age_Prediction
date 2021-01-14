from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QInputDialog, QLineEdit, QFileDialog,QLabel
from PyQt5.QtGui import QIcon,QPixmap, QImage,QPalette, QBrush
from PyQt5.QtCore import QSize

import os
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageOps

root='D:\\Proje 2'
model = load_model(root+'/Model_200x200x1_Cinsiyet.h5')
face_cascade=cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml')
Id_Gen = {0: 'Erkek', 1: 'Bayan'}
Gen_Id = dict((g, i) for i, g in Id_Gen.items())   
     
def goruntu():
    cap=cv2.VideoCapture(0)
    while True:
        ret , frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray ,1.3 ,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray=gray[y:y+h,x:x+w]
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
         
class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 500)
        #çv_box = QtWidgets.QVBoxLayout()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # buton resim
        self.Button_resim = QtWidgets.QPushButton(self.centralwidget)
        self.Button_resim.setGeometry(QtCore.QRect(140, 330, 120, 21))
        self.Button_resim.setObjectName("Button_resim")
        self.Button_resim.clicked.connect(self.resimYukle)
        self.Button_resim.setIcon(QIcon('C:/Users/OmerF/.spyder-py3/picture-icon.png'))
        self.Button_resim.setStyleSheet("color : black;""font-weight: bold")
        # buton kamera
        self.Button_kamera = QtWidgets.QPushButton(self.centralwidget)
        self.Button_kamera.setGeometry(QtCore.QRect(400, 330, 75, 23))
        self.Button_kamera.setObjectName("Button_kamera")
        self.Button_kamera.clicked.connect(self.kameraAc)
        self.Button_kamera.setIcon(QIcon('C:/Users/OmerF/.spyder-py3/camera-icon.png'))
        self.Button_kamera.setStyleSheet("color : black;""font-weight: bold")
        # grafik 
        self.image_frame = QtWidgets.QLabel()
        self.labelImage = QtWidgets.QLabel(self.centralwidget)
        self.labelImage.setGeometry(QtCore.QRect(50, 20, 256, 271))
        # sonuc yazılacak yaş
        self.yas = QtWidgets.QLabel(self.centralwidget)
        self.yas.setGeometry(QtCore.QRect(420, 70, 91, 16))
        self.yas.setText("")
        self.yas.setObjectName("yas")
         
        # text yaş
        self.sonuc_yas = QtWidgets.QLabel(self.centralwidget)
        self.sonuc_yas.setGeometry(QtCore.QRect(370, 70, 91, 16))
        self.sonuc_yas.setText("Yaş : ")
        self.sonuc_yas.setObjectName("sonuc_yas")
        self.sonuc_yas.setStyleSheet("color : black;""font-weight: bold") 
        # sonuc yazılacak cinsiyet
        self.cinsiyet = QtWidgets.QLabel(self.centralwidget)
        self.cinsiyet.setGeometry(QtCore.QRect(440, 110, 91, 16))
        self.cinsiyet.setText("")
        self.cinsiyet.setObjectName("cinsiyet")
        
        # text cinsiyet
        self.sonuc_cinsiyet = QtWidgets.QLabel(self.centralwidget)
        self.sonuc_cinsiyet.setGeometry(QtCore.QRect(370, 110, 91, 16))
        self.sonuc_cinsiyet.setText("Cinsiyet : ")
        self.sonuc_cinsiyet.setObjectName("sonuc_cinsiyet")
        self.sonuc_cinsiyet.setStyleSheet("color : black;""font-weight: bold") 
        # buton tahmin
        self.Button_calistir = QtWidgets.QPushButton(self.centralwidget)
        self.Button_calistir.setGeometry(QtCore.QRect(370, 190, 75, 23))
        self.Button_calistir.setObjectName("Button_resim")
        self.Button_calistir.clicked.connect(self.testEt)
        self.Button_calistir.setIcon(QIcon('C:/Users/OmerF/.spyder-py3/run-icon.png'))
        self.Button_calistir.setStyleSheet("color : black;""font-weight: bold")
      
        self.sonuc=''
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
       

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Form=MainWindow
        
    
    def resimYukle(self):
         options = QFileDialog.Options()
         options |= QFileDialog.DontUseNativeDialog
         self.fileName, _ = QFileDialog.getOpenFileName(self.Form, 'Open file','desktop',"Image Files (*.jpg *.gif *.bmp *.png)",options=options)
         
         pixmap = QPixmap(self.fileName)
         self.labelImage.setPixmap(pixmap)
         self.labelImage.setScaledContents(True);
         if self.fileName:
            print(self.fileName)   
         
         
         
     
    
    def testEt(self): 
        test_img = Image.open(str(self.fileName))
        test_img=ImageOps.grayscale(test_img)
        img = test_img.resize((200,200))
        img = np.array(img) / 255.0
        img = img.reshape(-1,200,200,1)
        age,race,gender = model.predict(img)
        yas=int(age[0]*100)
        self.sonuc=Id_Gen[gender[0].argmax()]
        print(gender[0].argmax())
        self.yas.setText(str(yas))
        self.cinsiyet.setText((self.sonuc))
        
        
        
    def kameraAc(self):
        #os.system('C:\\Users\\OmerF\\.spyder-py3\\Proje2_Deneme.py')
        goruntu()
            
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Button_resim.setText(_translate("MainWindow", "Fotoğraf Yükle"))
        self.Button_kamera.setText(_translate("MainWindow", "Kamera"))
        self.Button_calistir.setText(_translate("MainWindow", "Test Et"))
        
stylesheet = """
    QMainWindow {
        background-image: url("C:/Users/OmerF/.spyder-py3/filigran.jpg"); 
        background-repeat: no-repeat; 
        background-position: center;
    }
"""        
        

import sys
if __name__ == "__main__":    
 app = QtWidgets.QApplication(sys.argv)
 app.setStyleSheet(stylesheet)
 MainWindow = QtWidgets.QMainWindow()
 ui = Ui_MainWindow()
 ui.setupUi(MainWindow)
 MainWindow.show()
 sys.exit(app.exec_())

