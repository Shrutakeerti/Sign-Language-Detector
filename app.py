# app.py
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
model = tf.keras.models.load_model('sign_language_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        img = load_img(file, target_size=(64, 64))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        classes = list(train_generator.class_indices.keys())
        result = classes[class_idx]
        return result

if __name__ == '__main__':
    app.run(debug=True)
