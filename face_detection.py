import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

label = {"person_name": 1}

with open("lables.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while 1:

	#read frames
	ret, img = cap.read()
	#print(img)
	#print(ret)

	#convert to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		#draw rectangle around face
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		id1, pred = recognizer.predict(roi_gray)
		print(id1)
		print(labels[id1])

		font = cv2.FONT_HERSHEY_SIMPLEX
		name = labels[id1]
		color = (255, 125, 65)
		stroke = 2
		cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		img_item = 'img.png'
		cv2.imwrite(img_item, roi_gray)

		#detect eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)

		for(ex, ey, ew, eh) in eyes:
			#rectangle around eyes
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,127,0), 2)

	cv2.imshow('img', img)

	k = cv2.waitKey(30) & 0xff
	if(k == 27):
		break




cap.release()
cv2.destroyAllWindows()


