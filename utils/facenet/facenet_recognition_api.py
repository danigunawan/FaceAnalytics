import tensorflow as tf
from utils.face_detection import *
from utils.facenet import facenet
import numpy as np
import cv2


def load_facenet_model(model):
    facenet_graph = tf.Graph()
    with tf.io.gfile.GFile(model, 'rb') as f:
        with facenet_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return facenet_graph


facenet_graph = load_facenet_model('models/facenet_model/20180402-114759.pb')


def preprocess_for_facenet(images):
    """
    images: list of images
    return: list of cropped & prewhitened faces
    """
    cropped_prewhitened_aligned_faces = []

    for face_img in images:
        img_w, img_h, _ = face_img.shape
        face_location = get_face_bounding_boxes_dlib_hog(face_img)[0]
        x = face_location['x']
        y = face_location['y']
        w = face_location['w']
        h = face_location['h']
        crop_face = face_img[y:y + h - 1, x:x + w - 1]
        size = (160, 160)
        resized_face = cv2.resize(crop_face, size)
        prewhitened_face = facenet.prewhiten(resized_face)
        cropped_prewhitened_aligned_faces.append(prewhitened_face)

    return cropped_prewhitened_aligned_faces


def get_face_encodings(prewhitened_faces, facenet_graph):
    """
    faces: list of cropped prehitened faces
    return: list of face encodings
    """

    with tf.Session(graph=facenet_graph) as sess:
        # Get input and output tensors
        images_placeholder = facenet_graph.get_tensor_by_name("input:0")
        embeddings = facenet_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = facenet_graph.get_tensor_by_name("phase_train:0")
        feed_dict = {images_placeholder: prewhitened_faces, phase_train_placeholder: False}

        encodings = sess.run(embeddings, feed_dict=feed_dict)

    return encodings


def get_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    face_distance_list = []
    for face_encoding in face_encodings:
        face_distance_list.append(np.linalg.norm(face_encoding - face_to_compare))
    return face_distance_list


def get_face_names(faces, faces_info, tolerance, registered_face_encodings, registered_face_names):
    # preprocess face image for facenet
    preprocessed_faces = []
    for face in faces:
        size = (160, 160)
        resized_face = cv2.resize(face, size)
        prewhitened_face = facenet.prewhiten(resized_face)
        preprocessed_faces.append(prewhitened_face)

    # get face encoding
    face_encodings = get_face_encodings(preprocessed_faces, facenet_graph)

    # compare from database of face encoding
    for face_encoding, face_info in zip(face_encodings, faces_info):
        face_distances = get_face_distance(registered_face_encodings, face_encoding)
        print(face_distances)
        if min(face_distances) < tolerance:
            name = registered_face_names[np.argmin(face_distances)]

            # add name info to face info dictionary
            face_info['name'] = name
        else:
            face_info['name'] = 'unknown'

    return faces_info
