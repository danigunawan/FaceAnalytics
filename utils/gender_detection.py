from utils.SSRNET_model_age import SSR_net_general
import cv2
from config import model_path_configs
import numpy as np

# load model and weights
gender_model_weight_path = model_path_configs['gender_model_path']
img_size = 64
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
gender_classifier = SSR_net_general(img_size, stage_num, lambda_local, lambda_d)()
gender_classifier.load_weights(gender_model_weight_path)
gender_target_size = gender_classifier.input_shape[1:3]


def get_face_gender(faces, face_infos):
    # preprocess face images for gender model
    pre_processed_faces = np.array([cv2.resize(face, (img_size, img_size)) for face in faces])
    # print(pre_processed_faces.shape)

    # pass through model
    gender_predictions = gender_classifier.predict(pre_processed_faces)

    for gender_prediction, face_info in zip(gender_predictions, face_infos):
        if gender_prediction > 0.5:
            face_info['gender'] = 'M'
        else:
            face_info['gender'] = 'F'

    return face_infos
