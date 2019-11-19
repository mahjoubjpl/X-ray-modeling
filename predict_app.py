import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as k
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app=Flask(__name__)

def get_model():
    global model 
    model = load_model('model_binary_NIH.h5')
    print(" * Model loaded!")
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand-dims(image, axis=0)
    return image
print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])

def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(128, 128))
    
    prediction = model.predict(preprocess_image)
    
    response = {'prediction': {'Sick':prediction[0][0], 'NoSick':prediction[0][1]}}
            
    return jsonify(response)
