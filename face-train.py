import cv2
import os  #per le directories 
import numpy as np
from PIL import Image #per aprire l'immagine in una specifica directory
import pickle #l'alternativa al JSON per serializzare oggetti

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(
	cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizerLBPH = cv2.face.LBPHFaceRecognizer_create()
recognizerEigen= cv2.face.EigenFaceRecognizer_create()
recognizerFisher= cv2.face.FisherFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			pil_image = Image.open(path).convert("L")  # grayscale
			
			size = (400, 400)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			#final_image.show()
			
			image_array = np.array(final_image, "uint8")
			
			
			

			faces = face_cascade.detectMultiScale(
				image_array, scaleFactor=1.1, minNeighbors=6)

			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				roi_res = cv2.resize(roi, (350,350), interpolation=cv2.INTER_AREA)
				x_train.append(roi)
				y_labels.append(id_)


with open("pickles/face-labels.pickle", 'wb') as f: #creo il file .pickle
	pickle.dump(label_ids, f)						#ci scrivo all'interno la serializzazione di label_ids

recognizerLBPH.train(x_train, np.array(y_labels))
recognizerLBPH.save("recognizers/face-trainer-LBPH.yml")


# recognizerEigen.train(x_train, np.array(y_labels))
# recognizerEigen.save("recognizers/face-trainer-Eigen.yml")

# recognizerFisher.train(x_train, np.array(y_labels))
# recognizerFisher.save("recognizers/face-trainer-Fisher.yml")
