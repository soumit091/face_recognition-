import os
import numpy as np
from face_detect import extract_face
from embedder import get_embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

X, y = [], []
dataset_path = 'dataset/'

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        path = os.path.join(person_path, img_name)
        face = extract_face(path)
        if face is None:
            continue
        embedding = get_embedding(face)
        X.append(embedding)
        y.append(person)

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Train SVM
model = SVC(kernel='linear', probability=True)
model.fit(X, y_enc)

# Save model & label encoder
pickle.dump(model, open('models/svm_model.pkl', 'wb'))
pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))

print("Training complete. Model saved.")
