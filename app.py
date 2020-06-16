import sys
import os
import dlib
import glob
import numpy as np

if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "python app.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../data/picture1.jpg ../data/picture2.jpg\n")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
first_image = sys.argv[3]
second_image = sys.argv[4]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win1 = dlib.image_window()
win2 = dlib.image_window()

# Processing first image
print("Processing first image: {}".format(first_image))
img = dlib.load_rgb_image(first_image)
win1.clear_overlay()
win1.set_image(img)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
if len(dets) != 1:
    print("Detected more than one face in the first image\n")
    exit()
else:
    print("Detection coordinates for first image: Left: {} Top: {} Right: {} Bottom: {}".format(
        dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()))
    shape = sp(img, dets[0])
    win1.clear_overlay()
    win1.add_overlay(dets[0])
    win1.add_overlay(shape)
    face1_descriptor = facerec.compute_face_descriptor(img, shape)
print()

# Processing second image
print("Processing second image: {}".format(second_image))
img = dlib.load_rgb_image(second_image)
win2.clear_overlay()
win2.set_image(img)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
if len(dets) != 1:
    print("Detected more than one face in the second image\n")
    exit()
else:
    print("Detection coordinates for second image: Left: {} Top: {} Right: {} Bottom: {}".format(
        dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()))
    shape = sp(img, dets[0])
    win2.clear_overlay()
    win2.add_overlay(dets[0])
    win2.add_overlay(shape)
    face2_descriptor = facerec.compute_face_descriptor(img, shape)
print()

# Compute euclidean distance
first_face_vector = np.array(face1_descriptor)
second_face_vector = np.array(face2_descriptor)
print(type(first_face_vector), np.shape(first_face_vector))
print(type(second_face_vector), np.shape(second_face_vector))
dist = np.linalg.norm(first_face_vector - second_face_vector)
print("Euclidean Distance = ", dist)
print()

if dist < 0.6:
    print("Images are of the same person!")
else:
    print("Images are of different persons!")
print()

dlib.hit_enter_to_continue()
