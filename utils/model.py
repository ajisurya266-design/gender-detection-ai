import tensorflow as tf
import numpy as np
import cv2


# ===============================
# Load SavedModel
# ===============================
model = tf.saved_model.load(
    "gender_model_saved"
)

# Ambil fungsi inference
infer = model.signatures["serving_default"]


IMG_SIZE = 128

CLASS_NAMES = [
    "Perempuan",
    "Laki-laki"
]


def predict_gender(face):

    if face is None or face.size == 0:
        return "Tidak Diketahui", 0.0


    # Resize
    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )


    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    # Normalisasi
    face = face.astype("float32") / 255.0


    # Tambah batch
    face = np.expand_dims(
        face,
        axis=0
    )


    # ===============================
    # Prediksi SavedModel
    # ===============================

    input_tensor = tf.constant(face)

    result = infer(
        input_tensor
    )


    # Ambil output pertama
    prediction = list(result.values())[0].numpy()[0][0]


    # ===============================
    # Gender
    # ===============================

    if prediction >= 0.5:
        gender = CLASS_NAMES[1]
        confidence = prediction * 100

    else:
        gender = CLASS_NAMES[0]
        confidence = (1 - prediction) * 100


    return gender, float(confidence)