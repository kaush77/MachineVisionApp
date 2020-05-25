#importing libraries
import os
import numpy as np
import pickle
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for,jsonify,Response,make_response,current_app
from werkzeug.utils import secure_filename
import urllib.request

import cv2

#creating instance of the class
app=Flask(__name__)


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

IMAGE_SIZE = (256,256)
INPUT_SHAPE = (256,256,3)

@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():    

    if request.method == "POST":

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))             
            flash('Image successfully uploaded and displayed')

            # call prediction method
            img = load_image(filename) 
            result = predict(img)
 
            if '_' in result:
                result_array = result.split('_')
                session['p_class'] = result_array[0]
                session['p_confidence'] = result_array[1]
            else:
                session.pop('p_class', None)
                session.pop('p_confidence', None)

            return render_template('index.html', filename=filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    return render_template("index.html")

@app.route('/reset_index', methods=["GET"])
def reset_index():
    session.pop('p_class', None)
    session.pop('p_confidence', None)
    return redirect(url_for('index')) 

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

@app.route('/display/<filename>')
def display_image(filename):   
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
def load_image(filename):
    img = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255    
    return img

class_mapping = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

def predict(image):
    loaded_model = pickle.load(open("model/waste_image_classifier.pk","rb"))
    probabilities = loaded_model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)

    result = class_mapping[class_idx] + "_" + str(round(probabilities[class_idx],2))
    # return {class_mapping[class_idx]: probabilities[class_idx]}
    return result

 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)