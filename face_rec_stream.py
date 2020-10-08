import cv2
from imutils.video import WebcamVideoStream
import face_recognition
import os
import numpy as np
import imutils

names = []
imgpaths = []

data_path = r'/Users/timwu/custom_face_data'

for dirpath, dirname, filenames in os.walk(data_path):
    if dirpath != data_path:
        for filename in filenames:
            names.append(dirpath)
            imgpaths.append(os.path.join(dirpath,filename))

def get_boxes_encodings(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img, model='hog')
    encodings = np.array(face_recognition.face_encodings(img, boxes))
    return boxes, encodings

def most_common(lst):
    return max(set(lst), key=lst.count)

def compare_encodings(frame_enc, encodings, names):
    dist_list = np.array([np.linalg.norm(frame_enc-i) for i in encodings])
    idx = np.argsort(dist_list)[:5]
    names = list(np.array(names)[idx])
    return most_common(names)


names = [i.split('/')[-1] for i in names]
imgpaths = [i for i in imgpaths if "DS_Store" not in i]

encodings = [get_boxes_encodings(i)[-1] for i in imgpaths]
encodings = np.array(encodings)

vs = WebcamVideoStream(0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        boxes = face_recognition.face_locations(rgb,model='hog')
        encoding = face_recognition.face_encodings(rgb)

        for box, enc in zip(boxes, encoding):
            top, right, bottom, left = box
            print(right-left)
            cv2.rectangle(frame, (left, top),(right, bottom), color=(255,0,0), thickness=2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, compare_encodings(enc, encodings, names), (int((right+left)/2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow("Image", frame)

    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()