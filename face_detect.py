from mtcnn import MTCNN
import cv2

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    img = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if not results:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face
