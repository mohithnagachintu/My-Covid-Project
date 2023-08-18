import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('xception_model.h5')
print("Success")