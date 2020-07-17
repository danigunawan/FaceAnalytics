video_config = {
    "width": 640,
    "height": 480,
    "font_width": 2
}

model_path_configs = {
    "emotion_model_path": 'models/Emotion/model.onnx',
    "gender_model_path": 'models/Gender/SSRnet_model/ssrnet_3_3_3_64_1.0_1.0.h5'
}

frame_processor_config = {
    "skip_frame": 2,
    "bbox_color": (255, 0, 0),
    "bbox_text_color": (255, 255, 255)
}

age_model_config = {
    "weight_path": 'models/Age/ssrnet_3_3_3_64_1.0_1.0.h5'
}

face_recog_config = {
    "model_path": "models/facenet_model/20180402-114759.pb",
    "face_sim_tolerance": 0.9,
    "registered_face_path": "registered_faces_db/face_encodings/face_encoding_data.pickle"
}

face_detection_config = {
    "padding_ratio": 0.5
}
