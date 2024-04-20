from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import os
from ultralytics import YOLO
import time


app = Flask(__name__)
class_name = ['BANGUS','CATLA','MORI','MULLET','NEGATIVES','ROHU','SILVERCARP','SNAKEHEAD','TILAPIA']
model = load_model('model.h5')  


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_video(file_path):
    cap = cv2.VideoCapture(file_path)
    predictions = []

    if not cap.isOpened():
        return "Error: Failed to open video file."

    prediction_counts = {}

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (240, 240))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        p = model.predict(frame_rgb[np.newaxis, ...])
        predicted_class = class_name[np.argmax(p[0], axis=-1)]
        predictions.append(predicted_class)

        # Count the occurrences of each predicted class
        if predicted_class in prediction_counts:
            prediction_counts[predicted_class] += 1
        else:
            prediction_counts[predicted_class] = 1

    cap.release()

    # Find the prediction that occurred the most
    most_common_prediction = max(prediction_counts, key=prediction_counts.get)
    return [most_common_prediction], frame_rgb

@app.route("/")
def index():
    return render_template('index1.html')

@app.route("/fish-classifier")
def fish_classifier():
    return render_template('index.html')

@app.route("/fish-counting")
def fish_counting():
    return render_template('index2.html')

@app.route("/count",methods=["POST"])
def count():
    if request.method == "POST":
        model1 = YOLO('best.pt')
         
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template('index.html', message="No file part")

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file and allowed_file(file.filename):
            # Secure the filename before saving
            filename = secure_filename(file.filename)
            
            # Create the uploads directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            cap = cv2.VideoCapture(file_path)
            id = []
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    results = model1.track(frame,tracker = 'botsort.yaml', persist=True)
                    if results[0].boxes.id is None:
                        id.append(0.0)
                    else:
                        for i in results[0].boxes.id:
                            id.append(i.item())
                    x = max(id) if id else 0        
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows
        return render_template('index2.html', count=max(id))

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template('index.html', message="No file part")

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file and allowed_file(file.filename):
            # Secure the filename before saving
            filename = secure_filename(file.filename)
            
            # Create the uploads directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform predictions on the video
            predictions, frame_rgb = predict_video(file_path)

            # Save the frame as an image
            cv2.imwrite('static/frame.jpg', cv2.cvtColor(frame_rgb * 255, cv2.COLOR_RGB2BGR))

            return render_template('index.html', predictions=predictions)

if __name__=='__main__':
    app.run(host="0.0.0.0",port=int("8000"),debug=True)
