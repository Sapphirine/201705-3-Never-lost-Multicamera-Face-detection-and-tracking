from flask import Flask, render_template, Response, request, send_from_directory, redirect, url_for
import os, time, base64
from werkzeug import secure_filename
from datetime import datetime
import skvideo.io
import cv2
from Recognition import VideoProcess
from Camera import Camera

app = Flask(__name__)

app.config['PROJECT_PATH'] = '/home/paperspace/PycharmProjects/video-streaming'
app.config['UPLOAD_FOLDER'] = '/home/paperspace/PycharmProjects/video-streaming/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'jpg', 'avi', 'png'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

html = '''
{% extends "layout.html" %}

{% block content %}

    <!DOCTYPE html>
    <title>Upload File</title>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=upload>
    </form>
    
{% endblock %}
'''

def gen(camera):
    while True:
        frame = camera.getFrame()
        yield(b'--frame\r\n'
              b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def mainpage():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print file.filename
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileUrl = url_for('uploaded_file', filename=filename)
            video = VideoProcess(fileUrl)
            video.prepareDataset()
            video.run()
            downloadUrl = video.getOutput()
            return '<a href = ' + downloadUrl + '>Download</a>'

    return render_template(
        'upload.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/videofeed')
def videofeed():
    rootPath = 'uploads/'
    return Response(gen(Camera(rootPath + 'flash.avi')), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/manage')
def manage_file():
    filesList = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('manage.html', filesList=filesList)

@app.route('/delete/<filename>')
def delete_file(filename):
    filePath = app.config['UPLOAD_FOLDER'] + filename
    os.remove(filePath)
    return redirect(url_for('manage_file'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)