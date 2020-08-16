#https://www.youtube.com/watch?v=sz25xxF_AVE&t=574s
import cv2
import numpy as np
import face_recognition
imgBava = face_recognition.load_image_file("C:\\Users\\bavad\\Documents\\Python\\img\\bava.jpg")
imgBava = cv2.cvtColor(imgBava,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("C:\\Users\\bavad\\Documents\\Python\\img\\\Test.JPG")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
faceLoc =  face_recognition.face_locations(imgBava)[0]
encodeBava = face_recognition.face_encodings(imgBava)[0]
cv2.rectangle(imgBava,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest =  face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeBava],encodeTest)
faceDis = face_recognition.face_distance([encodeBava],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Bava bava',imgBava)
cv2.imshow('Bava Test',imgTest)
cv2.waitKey(0)