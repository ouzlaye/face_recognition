
from imutils.video import VideoStream, FPS
import numpy as np
import argparse, imutils, pickle, time
import cv2
import os

# construction des parseur 
ap = argparse.ArgumentParser()
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


# initialisation du streaming video
print("[INFO] demarrage de la vidéo...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

# on boucle sur les images du flux  vidéo
while True:
    frame = vs.read()
    #redimensionner le cadre pour avoir une largeur de 600 pixels
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    #creation du blob de l'image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Application du face detectorr de Openv CV
    detector.setInput(imageBlob)
    detections = detector.forward()

    # boucle pour la detection
    for i in range(0, detections.shape[2]):
        #extraire la confidence associée a la prediction
        confidence = detections[0, 0, i, 2]
        #filtrer les faibles detection
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7]* np.array([w , h, w, h])
            (startX, startY, endX, endY)= box.astype("int")

            # extraction de la face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # verifier si la taille de la face est suffisament large
            if fW <20 or fH <20 : 
                continue


            # Application de la face_recognition
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255,
            (96, 96), (0, 0, 0), swapRB=True, crop = False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # classifiacation 
            preds = recognizer.predict_proba(vec)[0]
            x = np.argmax(preds)
            proba = preds[x]
            name = le.classes_[x]

            # Afficher le bounding box + la probabilité
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY -10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
            cv2.FONT_ITALIC, 0.45, (0, 0, 255), 2)

    # update du compteur FPS
    fps.update()

    # Afficher la fentre video
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # quitter la boucle si on tappe sur la touche q
    if key == ord("q"):
        break

#arrêter la minuterie et afficher les informations FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
         
