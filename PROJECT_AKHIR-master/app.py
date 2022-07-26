import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import flask
from flask import Flask, render_template, Response, request, flash
from flask_socketio import SocketIO
from keras.models import load_model

from core_service.facerecognition import Recognizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qwerty123'

socketio = SocketIO(app)

PATH = '\\'.join(os.path.abspath(__file__).split('\\')[0:-1])
recognizer = Recognizer(
    socketio=socketio,
    facerecognition_model = os.path.join(PATH, "core_Service\\bin\\model-cnn-facerecognition.h5"), 
    labels_filename=os.path.join(PATH, "core_Service\\labels.csv"), 
    facedetection_model=os.path.join(PATH, "core_Service\\bin\\haarcascade_frontalface_default.xml"),
    use_mtcnn = False,
    camera_src=0
)

@app.route("/")
def index():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off':
        recognizer.close()
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on':
        recognizer.open()
        flash("Camera turn on!", "success")
    print("camera status", recognizer.status())
    return render_template("index.html", is_camera = recognizer.status())

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/face_registration")
def face_registration():
    return render_template("face_registration.html")
    
@app.route('/video_feed')
def video_feed():
    return Response(recognizer.gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)

    