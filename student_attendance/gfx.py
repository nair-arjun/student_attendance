"""Graphics utilities."""

import cv2

import numpy as np

from eigenfaces import EigenFaces

import csv

from datetime import date
import tensorflow

import keras

from keras.models import load_model
from keras.models import model_from_json

def load_emotion_model():
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("emotion_model.h5")
    return emotion_model


class Image:
    """Image manipulation class wrapping some opencv functions."""
    def __init__(self, ocv_image):
        """Wrap opencv image."""
        self._img = ocv_image
        self._face_cascade = 'haarcascade_frontalface_default.xml'

    def gray(self):
        """
        Returns:
            new grayscale image.
        """
        return Image(cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY))

    def faces(self):
        return map(lambda rect: self.cut(rect[0], rect[1], rect[2], rect[3]),
            self.face_areas())

    def cut(self, x, y, width, height):
        return Image(self._img[y:y + height, x:x + width])

    def scale(self, width, height):
        return Image(cv2.resize(self._img, (width, height)))

    def draw_rect(self, x, y, width, height):
        cv2.rectangle(self._img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    def save_to(self, path):
        cv2.imwrite(path, self._img)

    def to_numpy_array(self):
        return np.array(self._img, dtype=np.uint8).flatten()

    def put_text(self, text, x, y):
        cv2.putText(self._img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2,
            (0, 255, 0), 2)

    def face_areas(self):
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml') \
            .detectMultiScale(self._img, scaleFactor=1.2, minNeighbors=5,
                minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    def show(self):
        cv2.imshow('', self._img)
        cv2.waitKey(0)


def load_image_from(path):
    return Image(cv2.imread(path))


class FaceDetector:
    def __init__(self):
        self.clf = EigenFaces()
        self.clf.train('training_images')
        self.person_engagement = {}

    def save_engagement_data(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['person', 'attendance', 'total_time', 'engaged_time', 'confused_time', 'percentage_engaged', 'date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for person, engagement_data in self.person_engagement.items():
                total_time = engagement_data['total']
                engaged_time = engagement_data['engaged']
                confused_time = engagement_data['confused']
                percentage_engaged = engaged_time / total_time * 100
                writer.writerow({
                    'person': person,
                    'attendance': "present",
                    'total_time': total_time,
                    'engaged_time': engaged_time,
                    'confused_time': confused_time,
                    'percentage_engaged': percentage_engaged,
                    'date': date.today()
                })

    def show(self, image, wait=True):
        for (x, y, w, h) in image.face_areas():
            image.draw_rect(x, y, w, h)
            
            emotion_dict = {0: "Confused", 1: "Confused", 2: "Confused", 3: "Engaged", 4: "Engaged", 5: "Confused", 6: "Engaged"}
            emotion_model = load_emotion_model()

            face = image.cut(x, y, w, h).gray().scale(100, 100).to_numpy_array()
            face_new = image.to_numpy_array()
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face_new, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(emotion_prediction))
            emote = emotion_dict[maxindex]
            predicted_name = self.clf.predict_face(face)

            if predicted_name not in self.person_engagement:
                self.person_engagement[predicted_name] = {"engaged": 0, "confused": 0, "total": 0}
                
            self.person_engagement[predicted_name]["total"] += 1
                
            if emote == "Engaged":
                self.person_engagement[predicted_name]["engaged"] += 1
            else:
                self.person_engagement[predicted_name]["confused"] += 1

            for person in self.person_engagement:
                total_time = self.person_engagement[person]["total"]
                engaged_time = self.person_engagement[person]["engaged"]
                confused_time = self.person_engagement[person]["confused"]
                percentage_engaged = engaged_time / total_time * 100
            
            self.save_engagement_data('student_report.csv')

            image.put_text(predicted_name, x + 0, y + h + 30)
            image.put_text(emote, x + 100, y + h + 30)

        cv2.imshow('', image._img)
        if wait:
            cv2.waitKey(0)

    
