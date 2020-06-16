# Face Matching using DLib
Compute the 128D vector that describes the faces in each image identified by
shape. In general, if two face descriptor vectors have a Euclidean distance
between them less than 0.6 then they are from the same person, otherwise they
are from different people.

![input_example](https://github.com/moeabdol/face-matching/blob/master/example_input.png)

![output_example](https://github.com/moeabdol/face-matching/blob/master/example_output.png)

## Dependencies
* python3
* dlib
* numpy
* XQuartz

## How to run
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python appy.py ./model/shape_predictor_5_face_landmarks.dat ./model/dlib_face_recognition_resnet_model_v1.dat ./data/picture1.jpg ./data/picture2.jpg`
