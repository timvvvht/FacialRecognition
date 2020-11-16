# Facial Recognition
Facial Recognition is a python project that recognises faces from a webcam stream. 
![Face_Rec_gif](/media/facerec.gif)

The script uses the K-nearest neighbours algorithm to determine the closest match (using Euclidean distance as the metric of choice) of the faces identified in the webcam stream to encodings previously extracted from a database of images.

# Embedding Model
## The Easy Way
The <a href='face_rec_stream.py'>easy</a> way is to use the face_recognition library, which uses the dlib to predict high quality face embeddings with high precision. 

## The Hard(er) Way
The <a href ='Triplet_Loss/TripletLossModelTraining'>hard(er)</a> way is to train your own triplet loss model. For the embedding model, I used to transfer learning from a pre-trained Xception model in tensorflow, based on <a href='https://arxiv.org/pdf/1610.02357.pdf'>this paper</a>. The model was trained on the CelebA dataset, found <a href='http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'>here</a>, which contains ~200k images and ~10k identities. 

I then added a Global Max Pooling layer (performed much better than Global Average Pooling and Flatten) and a 2 fully connected layers afterwards. 

Each model was evaluated on my own dataset in this [notebook](Triplet_Loss/Triplet_Loss_model_eval.ipynb).

The best model was able to recognize correct identities on my own dataset, with a k-nearest neighbors algorithm with an accuracy of 85%. 

The model I trained performs reasonably well during the face-recognition from stream task, but was not as robust as using the face_recognition library. This was perhaps due to the limited amount of training data (which was already heavily augmented during the training process) whereas other triplet loss models such as <a href='https://arxiv.org/abs/1503.03832'>FaceNet</a> was trained on a much larger dataset. Moreover, the model I trained uses 96 x 96 size images for training, which may have slightly hampered the accuracy. Nonetheless, the real time prediction results perform well, with the exception of the occasional misidentification during blurry frames.  
  
# Face Recognition from stream 
1. Given a directory structure of ../database/identity/image.jpg, with at least K images per identity, run <a href='face_rec_embeddings.ipynb'> this script</a> to obtain 128-dimensional face encoding arrays for each image in the database. 

After obtaining face-encodings for each picture in the database, we will save the pandas dataframe containing such data into a pickle object for access in the live-facial recognition script.

2. We will then use cv2 and facial recognition library to identify faces in a video stream and find the 128-dimensional encodings for the faces found.

The encodings will then be compared with the ones in the pickled dataframe with encodings previously obtained. 

Using Euclidean distance as the metric of choice, we will find the K-nearest encodings (and their respective identities) with the encoding obtained from the video stream. 

Finally, we will output the highest frequency identity that are K-closest to the encoding read from the video stream. 

The script for the implementation with the face recognition library can be found <a href='face_rec_stream.py'>here</a>.
The script for the implementation with the custom trained model can be found [here](face_rec_stream_xception.py).
