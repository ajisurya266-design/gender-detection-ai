import tensorflow as tf

print(tf.__version__)

model = tf.keras.models.load_model(
    "gender_model.keras",
    compile=False
)

print("Model loaded")

model.export("gender_model_saved")

print("Export selesai")