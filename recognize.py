
import numpy as np
import argparse, imutils, pickle
import cv2
import os

#contruction de des arguments du parseurs 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


#lectures de faces serialiser depuis le dique 
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
protoPath1 = "deploy.prototxt"
modelPath = os.path.sep.join([args["detector"],
"res10_300x300_ssd_iter_140000.caffemodel"])
modelPath1 = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath1, modelPath1)

# lecture du face embedding model
embedding1= "openface_nn4.small2.v1.t7"
print("[INFO] loading face recognizer...")
#embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
embedder = cv2.dnn.readNetFromTorch(embedding1)

#lecture du modele de notre recognize
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# lecture d'une images pour la face_recognition
image = cv2.imread("images/n.jpg")
image = imutils.resize(image, width=600)
(h, w)= image.shape[:2]

#  creer un Blob depuis notre image
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

# appliquer le detecteur de face de Open CV 
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
    # extraire la confidence associé a la prediction
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extraction de la face ROI 
        face = image[startY: endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # verifier si la face est assez large
        if fW < 20 or fH< 20:
            continue


        faceBlob = cv2.dnn.blobFromImage(face, 1.0/ 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # effectuer une classification pour reconnaître le visage
        preds = recognizer.predict_proba(vec)[0]
        x = np.argmax(preds)
        proba = preds[x]
        name = le.classes_[x]


        # afficher le bounding box sur la face et la probalité qui est associée
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY -10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_ITALIC, 0.45, (0, 0, 255), 2)

# afficher la sortie sur une fenetre 
cv2.imshow("Image", image)
cv2.waitKey(0)

