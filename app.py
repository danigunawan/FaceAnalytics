# Import packages
import pafy
import cv2
from config import video_config
from modules.frame_processor import process_frame, draw_output_on_frame
from config import frame_processor_config

# Youtube Video URL
url = 'https://www.youtube.com/watch?v=wzSR2ZsvcH8'
    #'https://www.youtube.com/watch?v=vdl2lrAi5Xo'
    #'https://www.youtube.com/watch?v=EjK6SEui7uY'

# Get video Object
vPafy = pafy.new(url)
play = vPafy.getbest()

# Start the video
print(play.url)
cap = cv2.VideoCapture(play.url)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


frame_num = 0
skip_frame = frame_processor_config['skip_frame']
frame_info = {}

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    # Read frames
    ret, frame = cap.read()
    if ret:
        frame_num += 1

        # resize input image
        #frame = cv2.resize(frame, (video_config['width'], video_config['height']))

        if frame_num == 0 or frame_num % skip_frame == 0:
            # Pass through models
            # Face Detection + Face Recognition + Emotion Recognition + Gender + Age
            frame_info = process_frame(frame)

        # Draw Results on frame
        if frame_info:
            frame = draw_output_on_frame(frame, frame_info)

        # Display input & output image
        #frame = cv2.resize(frame, (video_config['width'], video_config['height']))


        cv2.imshow('input frame', frame)
        out.write(frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



cap.release()
out.release()
cv2.destroyAllWindows()
