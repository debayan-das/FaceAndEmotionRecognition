# importing libraries
import os
import cv2
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import img_to_array
from flask import Flask, render_template, Response, request
import csv
from PIL import Image

# load model
model = model_from_json(open("Resources/Data/fer.json", "r").read())
# load weights
model.load_weights('Resources/Datafer.h5')

face_app = Flask(__name__)
face_app.secret_key = 'face'
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# local Binary Pattern Histogram is an Face Recognizer algorithm
# inside OpenCV module used for training the image dataset
recognizer = cv2.face.LBPHFaceRecognizer_create()

# initializing global variables
cap = cv2.VideoCapture(0)
switch = 1
emotion = 0
face = 0
add = 0
face_id = ''
name = ''

# creating folders and files if not present
try:
    os.mkdir('./Resources/UserDetails')
except OSError as error:
    pass
try:
    os.mkdir('./Resources/TrainingImage')
except OSError as error:
    pass
try:
    os.mkdir('./Resources/TrainingImageLabel')
except OSError as error:
    pass

try:
    with open('Resources/UserDetails/UserDetails.csv', 'r') as file:
        pass
except IOError:
    with open('Resources/UserDetails/UserDetails.csv', 'a+') as file:
        wr = csv.writer(file)
        header = ['Id', 'Name']
        # entry of the header in csv file
        wr.writerow(header)
finally:
    # getting the name from "userdetails.csv"
    data_frame = pd.read_csv("Resources/UserDetails/UserDetails.csv")
    file.close()


def detect_emotion(test_img, gray_img, x, y, h, w):
    roi_gray = gray_img[y:y + h, x:x + w]  # cropping region of interest i.e. face area from  image
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('fear', 'disgust', 'angry', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.rectangle(test_img, (x, y - 20), (x + w, y), (255, 0, 0), -1)
    cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


def recognize_face(test_img, gray_img, x, y, h, w):
    try:
        recognized_id, conf = recognizer.predict(gray_img[y:y + h, x:x + w])
    except cv2.error:
        print('Model is not trained')
        return
    if conf < 50:
        recognized_name = data_frame.loc[data_frame['Id'] == recognized_id]['Name'].values
        if len(recognized_name) >= 1:
            recognized_name = recognized_name[0]
        detected_name = recognized_name + '-' + str(recognized_id)
    else:
        recognized_id = 'Mismatch'
        detected_name = str(recognized_id)
    cv2.putText(test_img, str(detected_name), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def get_faces_and_ids_from_directory(path):
    # get the path of all the files in the folder
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    # now looping through all the image paths and loading the ids and the faces saved in the folder
    for image_path in image_paths:
        # loading the image and converting it to gray scale
        pil_image = Image.open(image_path).convert('L')
        # now we are converting the PIL image into numpy array
        image_array = np.array(pil_image, 'uint8')
        # getting the id from the image
        id_of_image = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_array)
        ids.append(id_of_image)
    return faces, ids


def insert_data_and_train_faces():
    # creating the entry for the user in a csv file
    global data_frame
    row = [face_id, name]
    with open('Resources/UserDetails/UserDetails.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile)
        # Entry of the row in csv file
        writer.writerow(row)
    csvFile.close()

    # Saving the detected faces in variables
    faces, ids = get_faces_and_ids_from_directory("Resources/TrainingImage")
    # training the model with the captured image
    recognizer.train(faces, np.array(ids))
    # saving the trained model
    recognizer.save("Resources/TrainingImageLabel/Trainer.yml")
    # updating the data_frame
    data_frame = pd.read_csv("Resources/UserDetails/UserDetails.csv")


def gen_frames():
    global add, face
    sample_count = 0

    if face == 1:
        try:
            # reading the trained model
            recognizer.read("Resources/TrainingImageLabel/Trainer.yml")
        except cv2.error:
            face = 0
            print('Add some face')

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        test_img = cv2.flip(test_img, 1)
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        # it converts the images in different sizes (decreases by 1.32 times)
        # and 5 specifies the number of times scaling happens
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            # if detect emotion button is clicked
            if emotion == 1:
                detect_emotion(test_img, gray_img, x, y, h, w)

            # if recognize button is clicked
            if face == 1:
                recognize_face(test_img, gray_img, x, y, h, w)

            # if capture button is clicked
            if add == 1:
                sample_count += 1

                # saving the captured face in the dataset folder TrainingImage
                # as the image needs to be trained are saved in this folder
                cv2.imwrite(
                    "Resources/TrainingImage/ " + name + "." + face_id + '.' +
                    str(sample_count) + ".jpg", gray_img[y:y + h, x:x + w])

                if sample_count >= 60:
                    # after multiple captures
                    add = 0
                    print('Capture complete')
                    insert_data_and_train_faces()

        resized_img = cv2.resize(test_img, (1000, 700))
        if ret:
            try:
                ret, buffer = cv2.imencode('.jpg', resized_img)
                resized_img = buffer.tobytes()
                # return each frame to the img tag without breaking out of the loop
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + resized_img + b'\r\n')
            except Exception:
                pass
        else:
            pass


def is_number(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


@face_app.route('/')
def index():
    return render_template('index.html')


@face_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@face_app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, cap, emotion, face, add
    if request.method == 'POST':
        if request.form.get('stop') == 'Stop/Start Camera':
            if switch == 1:
                switch = 0
                emotion = 0
                face = 0
                add = 0
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(0)
                switch = 1

        if switch and request.form.get('emotion') == 'Detect Emotion':
            face = 0
            emotion = not emotion

        if switch and request.form.get('face') == 'Recognize Face':
            emotion = 0
            face = not face

        if switch and request.form.get('add') == 'Capture':
            global face_id, name
            add = not add
            face_id = str(len(data_frame) + 1)
            name = request.form.get('name')
            # checking if the face_id is numeric and name is Alphabetical
            if not (is_number(face_id) and name.isalpha()):
                print('Enter Numeric Id and Alphabetical Name')

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@face_app.route('/add_face')
def add_face():
    global emotion, face
    emotion = 0
    face = 0
    return render_template('addFace.html')


if __name__ == '__main__':
    face_app.run()

cap.release()
cv2.destroyAllWindows()
