import glob
import cv2
from utils.face_detection import get_face_bounding_boxes_dlib_hog
from utils.facenet.facenet_recognition_api import get_face_encodings, facenet_graph
from utils.facenet import facenet
import pickle
import os

face_encoding_data = {}
# Loop through images
for image_path in glob.glob('registered_faces_db/face_images/*.jpg'):
    face_name = os.path.basename(image_path).split('.')[0]
    image = cv2.imread(image_path)

    # Use face detection and face encoding to register faces
    face_bboxes = get_face_bounding_boxes_dlib_hog(image)
    x1, y1, x2, y2 = (face_bboxes[0]['x'], face_bboxes[0]['y'], face_bboxes[0]['x'] + face_bboxes[0]['w'],
                      face_bboxes[0]['y'] + face_bboxes[0]['h'])

    face_image = image[y1:y2, x1:x2]

    # preprocess for facenet
    size = (160, 160)
    resized_face = cv2.resize(face_image, size)
    prewhitened_face = facenet.prewhiten(resized_face)

    # face encoding
    face_encodings = get_face_encodings([prewhitened_face], facenet_graph)
    print(face_encodings[0])
    face_encoding_data[face_name] = face_encodings[0]

# save in DB/pickle file
pickle_out = open("registered_faces_db/face_encodings/" + "face_encoding_data.pickle", "wb")
pickle.dump(face_encoding_data, pickle_out)
pickle_out.close()

