#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import cv2
import sys
sys.path.append(os.path.abspath('../opencv'))
from PlateImage import PlateImage

### CONFIGURATION ###
app = Flask(__name__)

# folder, max img size
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

### Helper Functions ###

# gets full image path
def _gen_image_path(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

# applies opencv transformations to file
# returns path to transformed image
def _transform_img(image_path):
    # open image with opencv
    # apply transformations
    # save image at same path
    # return modified image (or unmodified if it bad)
    return image_path


### Routes ###

# determines whether a file name is valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_location)
            image = PlateImage(cv2.imread(upload_location))
            if not image.is_blurry():
                image.normalize_shape().save(path=upload_location)
                return redirect(url_for('uploaded_file', filename=filename))
            else:
                return render_template('error.html', error_msg="Image is too blurry. Please retake and upload a new image.")
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


### Run server ###
if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
