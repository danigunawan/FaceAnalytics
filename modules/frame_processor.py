from modules.face_analytics.face_analyzer import face_analyzer
import cv2
import numpy as np


def process_frame(cv2_img):
    """
    This function process image/frame and pass through various model and combine their results in a common JSON
    :param cv2_img: image to process
    :return: json response containing various model predictions
    """
    face_analytics_json = face_analyzer(cv2_img)

    return face_analytics_json


def draw_output_on_frame(cv2_img, json_info):
    """
    Overlay information(received from model prediction) on input image
    :param cv2_img:
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 0, 0)
    thickness = 2

    for face_info in json_info['face_analytics']:
        # Draw face bbox
        bbox = face_info.get("bbox", "")
        cv2_img = cv2.rectangle(cv2_img, (bbox['x'], bbox['y']), (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                                (255, 0, 0), thickness)

        # background for name
        label_size = cv2.getTextSize(face_info.get("name", "unknown"), font, font_scale, thickness=1)
        pts = np.array([[[bbox['x'], bbox['y']],
                         [int(bbox['x'] + label_size[0][0] + 5), int(bbox['y'])],
                         [int(bbox['x'] + label_size[0][0]), int(bbox['y'] - label_size[0][1] - 5)],
                         [bbox['x'], int(bbox['y'] - label_size[0][1] - 5)]
                         ]]
                       , dtype=np.int32)

        cv2.fillPoly(cv2_img, pts, (255, 0, 0))

        # overlay face name
        cv2_img = cv2.putText(cv2_img, face_info.get("name", "unknown"), (bbox['x'], bbox['y'] - 2), font, font_scale,
                              color=(255, 255, 255),
                              thickness=1)

        # overlay face emotion
        # background for emotion
        label_size = cv2.getTextSize(face_info.get("expression", "Neutral"), font, font_scale, thickness=1)
        pts = np.array([[[bbox['x'] + bbox['w'], bbox['y'] + bbox['h']],
                         [int(bbox['x'] + bbox['w'] - label_size[0][0] - 7), int(bbox['y'] + bbox['h'])],
                         [int(bbox['x'] + bbox['w'] - label_size[0][0] - 2),
                          int(bbox['y'] + bbox['h'] + label_size[0][1] + 4)],
                         [bbox['x'] + bbox['w'], int(bbox['y'] + bbox['h'] + label_size[0][1] + 2)]
                         ]]
                       , dtype=np.int32)

        cv2.fillPoly(cv2_img, pts, (255, 0, 0))

        # overlay face name
        cv2_img = cv2.putText(cv2_img, face_info.get("expression", "Neutral"),
                              (int(bbox['x'] + bbox['w'] - label_size[0][0]),
                               int(bbox['y'] + bbox['h'] + label_size[0][1] + 1)), font, font_scale,
                              color=(255, 255, 255),
                              thickness=1)

        # overlay face gender
        # background for name
        pts = np.array([[[int(bbox['x'] + bbox['w']), int(bbox['y'])],
                         [int(bbox['x'] + bbox['w']), int(bbox['y'] + 15)],
                         [int(bbox['x'] + bbox['w'] + 15), int(bbox['y'] + 15)],
                         [int(bbox['x'] + bbox['w'] + 15), int(bbox['y'])]
                         ]]
                       , dtype=np.int32)

        cv2.fillPoly(cv2_img, pts, (255, 0, 0))
        cv2_img = cv2.putText(cv2_img, face_info.get("gender", "X"), (bbox['x'] + bbox['w'] + 5, bbox['y'] + 12), font,
                              font_scale,
                              color=(255, 255, 255),
                              thickness=1)

        # overlay face age
        # background for age
        pts = np.array([[[int(bbox['x']), int(bbox['y'] + bbox['h'])],
                         [int(bbox['x'] - 15), int(bbox['y'] + bbox['h'])],
                         [int(bbox['x'] - 15), int(bbox['y'] + bbox['h'] - 15)],
                         [int(bbox['x']), int(bbox['y'] + bbox['h'] - 15)],
                         ]]
                       , dtype=np.int32)

        cv2.fillPoly(cv2_img, pts, (255, 0, 0))

        cv2_img = cv2.putText(cv2_img, str(int(face_info.get("age", "30"))),
                              (max(bbox['x'] - 14, 0), max(bbox['y'] + bbox['h'] - 5, 0)), font,
                              font_scale, color=(255, 255, 255), thickness=1)

    return cv2_img
