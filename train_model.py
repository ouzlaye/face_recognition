
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse, pickle

# Consttruct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

#load the face embedings 
print("[INFO] loading face embedings")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

#encode the labels 
print("[INFO] encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#train the model used to accept the 128-d embeddings ofthe face 
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel = "linear", probability=True)
recognizer.fit(data["embeddings"], labels)

#write the actual face recognition model to disk
f= open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

#write the label encodr to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()