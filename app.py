from flask import *
import os
import cv2
from werkzeug.utils import secure_filename
import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import PIL
from PIL.ImageDraw import Draw


MODEL_PATH = 'https://tfhub.dev/tensorflow/efficientdet/d0/1'
count=1

app = Flask(__name__)
UPLOAD_FOLDER = 'static/assets/uploads'
UPLOAD_Video = 'static/assets/uploads_video'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_Video'] = UPLOAD_Video
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','mp4'])
detector=""
camera=""
frame_count=1

@app.route("/")
def home():
    global detector,frame_count
    frame_count=1
    detector = hub.load(MODEL_PATH)
    
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_objects(path: str, model) -> dict:
    image_tensor = tf.image.decode_jpeg(
        tf.io.read_file(path), channels=3)[tf.newaxis, ...]
    return model(image_tensor)


def count_persons(path: str, model, threshold=0.23) -> int:
   
    results = detect_objects(path, model)
    # Class ID 1 = "person"
    return (results['detection_classes'].numpy()[0] == 1)[np.where(
        results['detection_scores'].numpy()[0] > threshold)].sum()


def draw_bboxes(image_path, data: dict, threshold=0.23) -> PIL.Image:
   
    image = PIL.Image.open(image_path)
    print(image)
    draw = Draw(image)

    im_width, im_height = image.size

    boxes = data['detection_boxes'].numpy()[0]
    classes = data['detection_classes'].numpy()[0]
    scores = data['detection_scores'].numpy()[0]

    for i in range(int(data['num_detections'][0])):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                      width=4, fill='red')

    return image

@app.route("/process_images",methods=['Get','Post'])
def process_images():

    global count,detector
    
  

    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    errors = {}
    success = False
    count_persons1 = 0

    if files[0] and allowed_file(files[0].filename):
        filename = secure_filename(files[0].filename)
        files[0].save(os.path.join(app.config['UPLOAD_FOLDER'], str(count)+"input.jpg"))   
        example_path = app.config['UPLOAD_FOLDER']+"/"+str(count)+"input.jpg"
        results = detect_objects(example_path, detector)
        count_persons1 =  count_persons(example_path,detector)
        image =  draw_bboxes(example_path, results)
        image.save(app.config['UPLOAD_FOLDER']+"/"+str(count)+"output.jpg")

        count+=1

    else:
        errors[files[0].filename] = 'File type is not allowed'





    
    return str(count_persons1)




@app.route("/process_Video",methods=['Get','Post'])
def process_Video():
    global camera
    camera = cv2.VideoCapture("http://192.168.21.144:5004/display/1.mp4")
    global count
    
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    errors = {}
    filename=""
    if files[0] and allowed_file(files[0].filename):
        filename = secure_filename(files[0].filename)
        files[0].save(os.path.join(app.config['UPLOAD_Video'],"1.mp4"))   

    else:
        errors[files[0].filename] = 'File type is not allowed'


    print(filename)
    return jsonify({
        'filename':filename
    })

def gen_frames():  
    global camera,detector,frame_count
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            cv2.imwrite('video/1.jpg',frame)
            results = detect_objects('video/1.jpg', detector)
    
            example_path='video/1.jpg'
            count_persons1 =  count_persons(example_path,detector)
            image =  draw_bboxes(example_path, results)
            
            image.save(example_path)

            frame=cv2.imread(example_path)

            status=""
            if count_persons1>10:
                status="Crowd people"
            else:
                status="None Crowd people"

            print("Frame: "+str(frame_count)+" count people: "+str(count_persons1)+" status: "+str(status))
            frame_count+=1
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display/<filename>')
def display_video(filename):
   

    
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='assets/uploads_video/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5004,debug=True)
    