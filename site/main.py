#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import imghdr
import os

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

def _generate_filename(filetype):
    alphanumeric = string.ascii_lowercase
    alphanumeric += string.digits
    # find an unused shortcode
    while True:
        chars = [random.choice(alphanumeric) for _ in range(6)]
        shortcode = ''.join(chars)
        c = db_cursor.execute("SELECT * FROM uploads"
                " WHERE shortcode = '%s'" % shortcode)
        # 0 rows so shortcode is unused
        if not c.fetchall():
            break
    return shortcode + '.' + filetype

# saves the image file to disk
def _save_image(input_file):
    filetype = imghdr.what(input_file)
    if filetype is None:
        return None

    filename = _generate_filename(filetype)
    path = _generate_path(filename)
    input_file.save(path)

    return path

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
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
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
    app.run(debug=True)
