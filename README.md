# Facial Recognition
Facial Recognition is a python project that recognises faces from a webcam stream. 
![Face_Rec_gif](/media/facerec.gif)

The script uses the K-nearest neighbours algorithm to determine the closest match (using Euclidean distance as the metric of choice) of the faces identified in the webcam stream to encodings previously extracted from a database of images.

# Usage 
1. Given a directory structure of ../database/identity/image.jpg, with at least K images per identity, run <a href='faWe wlce_rec_embeddings.ipynb'> this script</a> to obtain 128-dimensional face encoding arrays for each image in the database. 

The most straight-forward approach for this task is to utilise the <a href='https://pypi.org/project/face-recognition/'>face recognition library</a>. 

After obtaining face-encodings for each picture in the database, we will save the pandas dataframe containing such data into a pickle object for access in the live-facial recognition script.

2. We will then use cv2 and facial recognition library to identify faces in a video stream and find the 128-dimensional encodings for the faces found.

The encodings will then be compared with the ones in the pickled dataframe with encodings previously obtained. 

Using Euclidean distance as the metric of choice, we will find the K-nearest encodings (and their respective identities) with the encoding obtained from the video stream. 

Finally, we will output the highest frequency identity that are K-closest to the encoding read from the video stream. 

The script for this implementation can be found <a href='face_rec_stream.py'>here</a>.
