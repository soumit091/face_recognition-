import cv2
from face_detect import extract_face
from embedder import get_embedding
import pickle

# Load models
model = pickle.load(open('models/svm_model.pkl', 'rb'))
label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

# Load and process test image
img_path = 'test.jpg'  # replace with your test image
face = extract_face(img_path)
if face is None:
    print("No face detected.")
else:
    embedding = get_embedding(face)
    pred_class = model.predict([embedding])[0]
    pred_prob = model.predict_proba([embedding])[0][pred_class]
    person = label_encoder.inverse_transform([pred_class])[0]
    print(f"Predicted: {person} ({pred_prob:.2f} confidence)")
