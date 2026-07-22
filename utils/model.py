import tensorflow as tf
import numpy as np
import cv2

# ===============================
# Load Model (sekali saja)
# ===============================
model = tf.keras.models.load_model(
    "gender_model.h5",
    compile=False
)

IMG_SIZE = 128

# Sesuaikan dengan folder dataset saat training
# Jika train/
# ├── female
# └── male
#
# maka:
CLASS_NAMES = ["Perempuan", "Laki-laki"]


def predict_gender(face):

    if face is None or face.size == 0:
        return "Tidak Diketahui", 0.0

    # Resize
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    # BGR -> RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Normalisasi
    face = face.astype("float32") / 255.0

    # Tambah dimensi batch
    face = np.expand_dims(face, axis=0)

    # Prediksi
    prediction = model.predict(face, verbose=0)[0][0]

    # Karena output sigmoid
    if prediction >= 0.5:
        gender = CLASS_NAMES[1]
        confidence = prediction * 100
    else:
        gender = CLASS_NAMES[0]
        confidence = (1 - prediction) * 100

    return gender, float(confidence)