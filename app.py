# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, render_template, request
from keras.utils import img_to_array, load_img
from keras.models import load_model
import os
import numpy as np
from keras import layers


app = Flask(__name__)
app=Flask(__name__,template_folder='template')
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_FOLDER, 'D:/Potato_Classification/model/1')
model = load_model(MODEL_PATH)
IMAGE_SIZE = 256
IMAGES_FOLDER = os.path.join('static','images')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class_indices = {0:'early_blight',1:'Late Blight', 2:'healthy'}
class_names = ['early_blight', 'healthy', 'late_blight']
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('base.html')


@app.route('/', methods=['POST'])
def predict():
    
    for f in os.listdir('static/images'):
        os.remove(os.path.join('static/images',f))
    imagefile = request.files['imagefile']

    if(not imagefile):
        return render_template('base.html', nofile="error")
    
    if(not allowed_file(imagefile.filename)):
        return render_template('base.html', notimage="error")

    
    full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(full_image_path)
    image = load_img(full_image_path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255

    pred = model.predict(image)
    print(pred)
    predicted_class = class_indices[np.argmax(pred)]


    return render_template('base.html', image=full_image_path, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
