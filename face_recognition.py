import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(' ', '_').lower()

			#encoding labels and convert file to np array
			if label not in label_ids:
				label_ids[label] = current_id
				current_id += 1

			#cant use id so id1
			id1 = label_ids[label]

			pil_image = Image.open(path).convert("L")#L for gray scale
			size = (512, 512)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")

			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array, 1.5, 10)
			#print(faces)

			for(x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				print(roi)
				x_train.append(roi)
				y_labels.append(id1)

print(label_ids)

with open("lables.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

