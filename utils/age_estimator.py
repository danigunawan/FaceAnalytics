import cv2
import numpy as np
from utils.SSRNET_model_age import SSR_net
from config import age_model_config

# load model and weights
img_size = 64
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
model.load_weights(age_model_config['weight_path'])


def get_face_age(faces, faces_info):

    # resize for model
    faces = [cv2.resize(face, (64, 64)) for face in faces]

    # change color
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]

    # pass through model
    faces = np.array(faces)
    predicted_ages = model.predict(faces)

    for face_info, predicted_age in zip(faces_info, predicted_ages[0]):
        face_info['age'] = predicted_age
    return faces_info
