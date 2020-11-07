import cv2
from imutils.video import WebcamVideoStream
from tensorflow.keras.applications import Xception
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import numpy as np
import imutils
import pandas as pd
from mtcnn.mtcnn import MTCNN

# Reads the pandas dataframe of face encodings that was previously extracted
df = pd.read_pickle('custom_embeddings_xception.pkl')
weights = r'Triplet Loss/global_max_pooling_FC_4_1_xception.0.4527.hdf5'

names = df.Name.tolist()
encodings = df.Encodings.tolist()
detector = MTCNN()


def load_model(weights):
    input_shape = (96, 96, 3)
    embedding_size = 128

    xception = Xception(weights="imagenet", input_shape=input_shape, include_top=False)
    xception.trainable = False

    inputs = Input(shape=input_shape)
    # Layer for Xception preprocessing
    layer = Lambda(lambda x: (x / 127.5) - 1)(inputs)
    layer = xception(layer)
    layer = GlobalMaxPooling2D()(layer)
    layer = Dense(embedding_size * 4, activation='relu')(layer)
    layer = Dense(embedding_size)(layer)
    layer = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(layer)
    model = Model(inputs, layer)
    model.load_weights(weights)
    return model


def img_to_emb(img_array, embedding_model):
    return embedding_model.predict(np.expand_dims(img_array, axis=0))


def extract_face_coords(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f = detector.detect_faces(img)
    left, top, w, h = f[0]['box']
    left, top = abs(left), abs(top)
    right = abs(left+w)
    bottom = abs(top+h)
    return left, top, right, bottom


def crop_to_face(img, left, top, right, bottom):
    face = img[top:bottom, left:right]
    face = cv2.resize(face, (96,96))
    return face

def most_common(array):
    array = list(array)
    return max(set(array), key=array.count)


def knn(frame_enc, encodings, names, neighbors=5):
    dist_list = np.array([np.linalg.norm(frame_enc - i) for i in encodings])
    dist_list = np.array([i for i in dist_list if i < 1])
    try:
        idx = np.argsort(dist_list)[:neighbors]
        names = np.array(names)[idx]
        print(dist_list[idx])
        print(names)
        identity = most_common(names)
        if len(set(names)) < 5 and identity == names[0]:
            print(identity)
            return identity

    except ValueError:
        pass


model = load_model(weights)
vs = WebcamVideoStream(0).start()
frames = 0
while True:
    frames += 1
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = detector.detect_faces(img)
    boxes = []
    encoding = []

    for i in f:
        left, top, w, h = i['box']
        left, top = abs(left), abs(top)
        right = abs(left + w)
        bottom = abs(top + h)
        boxes.append((left, top, right, bottom))
        cropped = crop_to_face(img, left, top, right, bottom)
        encoding.append(img_to_emb(cropped, model))

    for box, enc in zip(boxes, encoding):
        left, top, right, bottom = box
        cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, knn(enc, encodings, names, neighbors=5), (int((left + right) / 2), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.imshow("Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

