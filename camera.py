import cv2
import numpy as np
from detection import AccidentDetectionModel

model_path = "/content/drive/MyDrive/AccDetection/model.json"
weights_path = "/content/model_weights.h5"
model = AccidentDetectionModel(model_path, weights_path)

font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication(video_path):
    video = cv2.VideoCapture(video_path)  # Use video file path
    if not video.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video stream.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :, :])  # Add fourth dimension for batch size
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)
            
            # to beep when alert:
            # if prob > 90:
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        
        cv2.imshow('Video', frame)
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = '/content/Accident Kegalle (CCTV Record By Gils Techno Solution(Pvt)Ltd.2019.02.24.mp4'  # Replace with the name of your uploaded video file
    startapplication(video_path)
