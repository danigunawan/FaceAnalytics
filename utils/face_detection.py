import dlib
import cv2

cnn_face_detector_path = 'models/dlib_cnn_face_detector/mmod_human_face_detector.dat'
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)


def get_face_bounding_boxes_dlib_hog(frame):
    rects = hog_face_detector(frame, 1)
    face_locations = []
    for i, d in enumerate(rects):
        x = d.left()
        y = d.top()
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        face_locations.append({'x': x, 'y': y, 'w': w, 'h': h})

    return face_locations


def get_face_bounding_boxes_dlib_cnn(frame):
    rects = cnn_face_detector(frame, 1)
    face_locations = []
    for i, d in enumerate(rects):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - d.rect.left()
        h = d.rect.bottom() - d.rect.top()
        face_locations.append({'x': x, 'y': y, 'w': w, 'h': h})

    return face_locations


def add_face_padding(face_location, img_w, img_h):
    ad = 0.5
    x1 = face_location['x']
    y1 = face_location['y']
    x2 = face_location['x'] + face_location['w']
    y2 = face_location['y'] + face_location['h']
    w = x2 - x1
    h = y2 - y1
    xw1 = max(int(x1 - ad * w), 0)
    yw1 = max(int(y1 - ad * h), 0)
    xw2 = min(int(x2 + ad * w), img_w - 2)
    yw2 = min(int(y2 + ad * h), img_h - 2)

    modified_face_location = {'x': xw1, 'y': yw1, 'w': xw2 - xw1, 'h': yw2 - yw1}

    return modified_face_location


def draw_bounding_boxes_dlib(frame, face_locations):
    for i, face_location in enumerate(face_locations):
        x1 = face_location['x']
        y1 = face_location['y']
        x2 = face_location['x'] + face_location['w']
        y2 = face_location['y'] + face_location['h']
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0, 155, 255),
                      2)
    return frame


def convert_json_face_locations_into_dlib_rectangle(face_locations):
    rects = []
    for face_location in face_locations:
        x1 = face_location['x']
        y1 = face_location['y']
        x2 = face_location['x'] + face_location['w']
        y2 = face_location['y'] + face_location['h']

        # (top, right, bottom, left)
        rects.append(dlib.rectangle(x1, y1, x2, y2))
    return rects
