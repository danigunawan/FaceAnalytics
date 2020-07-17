import onnxruntime as rt
import numpy as np
import cv2
import config

sess = rt.InferenceSession(config.model_path_configs['emotion_model_path'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


def rgb2gray(rgb):
    """Convert the input image into grayscale"""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def resize_img(img_to_resize):
    """Resize image to FER+ model input dimensions"""
    r_img = cv2.resize(img_to_resize, dsize=(64, 64), interpolation=cv2.INTER_AREA)
    r_img.resize((1, 1, 64, 64))
    return r_img


def preprocess(img_to_preprocess):
    """Resize input images and convert them to grayscale."""
    if img_to_preprocess.shape == (64, 64):
        img_to_preprocess.resize((1, 1, 64, 64))
        return img_to_preprocess

    grayscale = rgb2gray(img_to_preprocess)
    processed_img = resize_img(grayscale)
    return processed_img


def detect_face_emotion(frame, face_locations):
    faces_emotions = []
    emotion_labels = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']
    for i, face_location in enumerate(face_locations):
        x = face_location['x']
        y = face_location['y']
        w = face_location['w']
        h = face_location['h']
        face_img = frame[y:y+h, x:x+w]
        x = preprocess(face_img)
        x = x.astype(np.float32)
        pred_onnx = sess.run(None, {input_name: x})
        emotion_result = emotion_labels[np.argmax(pred_onnx[0][0])]
        faces_emotions.append(emotion_result)

    return faces_emotions


def write_emotion_on_frame(frame, face_emotions, face_locations):

    for face_emotion, face_location in zip(face_emotions, face_locations):
        x = face_location['x']
        y = face_location['y']
        w = face_location['w']
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, face_emotion, (x, y - 2), font, w/150, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
