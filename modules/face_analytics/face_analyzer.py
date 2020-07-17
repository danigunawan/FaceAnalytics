from utils.face_detection import get_face_bounding_boxes_dlib_hog, add_face_padding
from utils.emotion_detection import preprocess as emotion_preprocess, sess, input_name
from utils.age_estimator import get_face_age
from utils.facenet.facenet_recognition_api import get_face_names
from utils.gender_detection import get_face_gender
import numpy as np
from config import face_recog_config, face_detection_config
import pickle

with open(face_recog_config['registered_face_path'], "rb") as file:
    registered_face_data = pickle.load(file)
    registered_face_names = list(registered_face_data.keys())
    registered_face_encodings = list(registered_face_data.values())

face_sim_tolerance = face_recog_config['face_sim_tolerance']


def face_analyzer(cv2_img):
    face_info = []
    face_images = []

    # face Detection - Get face bounding boxes
    face_bboxes = get_face_bounding_boxes_dlib_hog(cv2_img)

    img_h, img_w, _ = cv2_img.shape

    # FOR EACH FACE -------
    padded_faces = []
    for face_bbox in face_bboxes:
        # add padding to face image
        paddedd_face_bbox = add_face_padding(face_bbox, img_w, img_h)

        face_info.append({'bbox': face_bbox})

        face_images.append(
            cv2_img[face_bbox['y']:face_bbox['y'] + face_bbox['h'], face_bbox['x']: face_bbox['x'] + face_bbox['w']])

        padded_faces.append(cv2_img[paddedd_face_bbox['y']:paddedd_face_bbox['y'] + paddedd_face_bbox['h'],
                            paddedd_face_bbox['x']: paddedd_face_bbox['x'] + paddedd_face_bbox['w']])

    if len(face_bboxes) > 0:
        # face Recognition  - Get face Names
        face_info = get_face_names(face_images, face_info, face_sim_tolerance, registered_face_encodings,
                                   registered_face_names)

        # face Emotion Prediction - get face Emotion
        face_info = get_face_emotion(face_images, face_info)

        # if padded_faces is not None:
        # face Age Estimation - get Age Estimation
        face_info = get_face_age(padded_faces, face_info)

        # face Gender Prediction - Get Gender from face
        face_info = get_face_gender(padded_faces, face_info)

    return {"face_analytics": face_info}


def get_face_emotion(face_images, faces_info):
    emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']
    face_images = np.array([emotion_preprocess(face_image) for face_image in face_images])
    face_images = face_images.astype(np.float32)

    for face_image, face_info in zip(face_images, faces_info):
        pred_onnx = sess.run(None, {input_name: face_image})
        face_expression = emotion_labels[np.argmax(pred_onnx[0][0])]
        face_info['expression'] = face_expression

    return faces_info
