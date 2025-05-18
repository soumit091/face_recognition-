from keras.models import load_model
import numpy as np

model = load_model('facenet_keras.h5')

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    return model.predict(samples)[0]
