import tensorflow as tf

model = tf.keras.models.load_model(
    "gender_model.h5",
    compile=False
)

print("Model berhasil dibuka!")
model.summary()