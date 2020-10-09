import cv2
from imutils.video import WebcamVideoStream
import face_recognition
import numpy as np
import imutils
import pandas as pd

df = pd.read_pickle('custom_embeddings.pkl')

names = df.Name.tolist()
encodings = df.Encodings.tolist()


def get_boxes_encodings(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img, model='hog')
    encodings = np.array(face_recognition.face_encodings(img, boxes))
    return boxes, encodings


def most_common(array):
    array = list(array)
    return max(set(array), key=array.count)


def knn(frame_enc, encodings, names, neighbors=5):
    dist_list = np.array([np.linalg.norm(frame_enc - i) for i in encodings])
    idx = np.argsort(dist_list)[:neighbors]
    names = np.array(names)[idx]
    return most_common(names)


vs = WebcamVideoStream(0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='hog')
    encoding = face_recognition.face_encodings(rgb)

    for box, enc in zip(boxes, encoding):
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, knn(enc, encodings, names), (int((right + left) / 2), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.imshow("Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
