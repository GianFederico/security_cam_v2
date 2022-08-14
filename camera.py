import cv2
import time
from cv2 import imshow
import numpy as np
import pickle

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profileface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainer-LBPH.yml")
# recognizer= cv2.face.EigenFaceRecognizer_create()
# recognizer.read("recognizers/face-trainer-Eigen.yml")
# recognizer= cv2.face.FisherFaceRecognizer_create()
# recognizer.read("recognizers/face-trainer-Fisher.yml")


labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v: k for k, v in og_labels.items()}

detection = False
detection_stopped_time = None
timer_started = False
# voglio registrare 3 secondi dopo che non detecto più nulla
SECONDS_TO_RECORD_AFTER_DETECTION = 3
# altrimenti avrò un sacco di video da pochi millisecondi
angle = 0

# width e height of cap in float, quindi convertita in int
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

i = 0
while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1.3 affects la velocità d'esecuzione a discapito dell'accuracy (più è basso più è accurate)
    faces = face_cascade.detectMultiScale(gray, 1.2, 12)
    profilefaces = profileface_cascade.detectMultiScale(gray, 1.2, 5)
    #4 sto dicendo che mi serve identificare almeno 4 facce nel neighborhood per identificarne effettivamente 1

    if len(faces)+len(profilefaces) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            out = cv2.VideoWriter(
                f"face_detected_at_frame_{i}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")
    elif detection:  # abbiamo identificato qualcosa prima, ma ora non più, quindi dobbiamo aspettare 3 sec
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if len(faces) == 0:
        for (x, y, width, height) in profilefaces:
            cv2.rectangle(frame, (x, y), (x + width, y + height),
                (255, 255, 255), 3)
    else:

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height),
                        (255, 255, 255), 3)
        roi_gray = gray[y:y+height, x:x+width]


        # eyes = eye_cascade.detectMultiScale(gray[y:(y + height), x:(x + width)], 1.4, 5)
        # index = 0
        # eye_1 = [None, None, None, None]
        # eye_2 = [None, None, None, None]
        # for (ex, ey, ew, eh) in eyes:
        #         if index == 0:
        #             eye_1 = [ex, ey, ew, eh]
        #         elif index == 1:
        #             eye_2 = [ex, ey, ew, eh]
        #         cv2.rectangle(frame[y:(y + height), x:(x + width)], (ex, ey),
        #                     (ex + ew, ey + eh), (0, 255, 0), 1)
        #         index = index + 1
        #         if (eye_1[0] is not None) and (eye_2[0] is not None):
        #             if eye_1[0] < eye_2[0]:
        #                 left_eye = eye_1
        #                 right_eye = eye_2
        #             else:
        #                 left_eye = eye_2
        #                 right_eye = eye_1
        #             left_eye_center = (
        #                 int(left_eye[0] + (left_eye[2] / 2)),
        #                 int(left_eye[1] + (left_eye[3] / 2)))

        #             right_eye_center = (
        #                 int(right_eye[0] + (right_eye[2] / 2)),
        #                 int(right_eye[1] + (right_eye[3] / 2)))

        #             left_eye_x = left_eye_center[0]
        #             left_eye_y = left_eye_center[1]
        #             right_eye_x = right_eye_center[0]
        #             right_eye_y = right_eye_center[1]

        #             delta_x = right_eye_x - left_eye_x
        #             delta_y = right_eye_y - left_eye_y

        #             # Slope of line formula
        #             if delta_x == 0:
        #                 angle = np.arctan(delta_y / (delta_x+0.0001))
        #             else:
        #                 angle = np.arctan(delta_y / delta_x)

        #             # Converting radians to degrees
        #             angle = (angle * 180) / np.pi

        #             if angle > 5:
        #                 cv2.putText(frame, 'right tilt: ' + str(int(angle))+'deg.',
        #                     (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        #                     (255, 255, 255), 2, cv2.LINE_AA)

        #                 image_center = tuple(np.array(roi_gray.shape[1::-1]) / 2)
        #                 rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        #                 result = cv2.warpAffine(roi_gray, rot_mat, roi_gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        #                 roi_gray=result
        #             elif angle < -5:
        #                 cv2.putText(frame, 'left tilt: ' + str(int(angle))+'deg.',
        #                         (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        #                         (255, 255, 255), 2, cv2.LINE_AA)

        #                 image_center = tuple(np.array(roi_gray.shape[1::-1]) / 2)
        #                 rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        #                 result = cv2.warpAffine(roi_gray, rot_mat, roi_gray.shape[1::-1], flags=cv2.INTER_LINEAR)
        #                 roi_gray=result

        #roi_gray = cv2.resize(roi_gray, (250,250), interpolation=cv2.INTER_AREA)
        imshow("roi", roi_gray)  


        id_, conf = recognizer.predict(roi_gray) #conf = distance of the face from the training set (chi-sqr)
        if conf >= 4 and conf <= 75:
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            acc= 100-(float(conf))
            cv2.putText(frame, name + ' ' + str(round(acc,2)) + '%', (x, y-10), font,
                        0.8, color, stroke, cv2.LINE_AA)
        else:
            name = "unknown"
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y-10), font,
                        0.8, color, stroke, cv2.LINE_AA)

    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    i = i+1

out.release()
cap.release()
cv2.destroyAllWindows()
