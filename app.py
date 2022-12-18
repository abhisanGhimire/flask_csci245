# sabai import garyo
from flask import Flask,render_template,Response,request
import cv2
import json
import os
import shutil
import numpy as np
import datetime
import face_recognition

app=Flask(__name__)
# # camera le capture garyo
camera=cv2.VideoCapture(0)
previous_date_time=datetime.date.today()
student_present=[]
total_students=os.listdir("student_images")
print(total_students)





# JUST FOR NOME PAGE-------------------------------------------------------------------------------------------------------------

# euta function frame generate garna
def generate_frames():
    # infinite running loop
    while True:
        ## read the camera frame
        success,frame=camera.read()
        # success bhayo bhaney
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/harcascade_frontalface.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            if isinstance(faces, np.ndarray):
                print("true")
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/testroute')
def testroute():
    return "Working"

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# JUST FOR NOME PAGE-------------------------------------------------------------------------------------------------------------






#TRAIN DATA---------------------------------------------------------------------------------------------------------------------

@app.route('/train_data')
def train_data():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            return json.dumps({"result":"failed","position":"record_image:camera read error"})
        else:
            detector=cv2.CascadeClassifier('Haarcascades/harcascade_frontalface.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            if isinstance(faces, np.ndarray):
                    img_name = "_temporary_.png"
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    return json.dumps({"result":"success","image_saved":"true"})
            else:
                return json.dumps({"result":"failed"})

#TRAIN DATA---------------------------------------------------------------------------------------------------------------------




#UPDATE NAME-----------------------------------------------------------------------------------------------------------------------
@app.route('/update_name')
def update_name():
    student_name=request.args.get('name')
    if(student_name== None):
        return json.dumps({"Result":"invalid name"})
    else:
        file=os.getcwd()+'\_temporary_.png'    
        # student_images folder chaina bhaney banaera current working directory change garcha
        if(not os.path.isdir("student_images")):
            os.mkdir("student_images")
            os.chdir("student_images")
        else:
            os.chdir("student_images")
        
        if(not os.path.isdir(student_name)):
            os.mkdir(student_name)
            os.chdir(student_name)
            print(os.getcwd())
            shutil.copy2(file,student_name+".png")
            os.chdir("..")
        else:
            os.chdir("..")
            return json.dumps({"result":"student name exists"})
    os.chdir("..")
    print(os.getcwd())
    return json.dumps({"result":"success"})
#UPDATE NAME-----------------------------------------------------------------------------------------------------------------------



#TAKE ATTENDANCE-------------------------------------------------------------------------------------------------------------------
@app.route('/take_attendance')
def take_attendance():
    print(os.getcwd())
    global previous_date_time,student_present, total_students
    today_date_time = datetime.date.today()
    if today_date_time==previous_date_time:
        previous_date_time=today_date_time
    elif today_date_time!=previous_date_time:
        student_present=[]
        previous_date_time=today_date_time

    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    #Load a sample picture and learn how to recognize it.
    os.chdir("student_images")
    for file in os.listdir():
        print(file)
        os.chdir(file)
        known_face_names.append(file)
        image=face_recognition.load_image_file(file+".png")
        known_face_encodings.append(face_recognition.face_encodings(image)[0])
        os.chdir("..")
        # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = frame
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            for face in face_names:
                if student_present==[] and face!="Unknown":
                    student_present.append(face)
                    missing_student=list(set(total_students)-set(student_present))
                    os.chdir("..")
                    print(os.getcwd())
                    return json.dumps({"result":"success","new_name":face,"total_student":str(len(student_present)),"student_names":student_present,"missing_student":missing_student})
                elif face not in student_present and face!="Unknown":
                    student_present.append(face)
                    missing_student=list(set(total_students)-set(student_present))
                    os.chdir("..")
                    print(os.getcwd())
                    return json.dumps({"result":"success","new_name":face,"total_student":str(len(student_present)),"student_names":student_present,"missing_student":missing_student})
                elif face=="Unknown":
                    os.chdir("..")
                    print(os.getcwd())
                    missing_student=list(set(total_students)-set(student_present))
                    return json.dumps({"result":"unknown","new_name":face,"total_student":str(len(student_present)),"student_names":student_present,"missing_student":missing_student})
                elif face in student_present:
                    os.chdir("..")
                    print(os.getcwd())
                    missing_student=list(set(total_students)-set(student_present))
                    return json.dumps({"result":"already","new_name":face,"total_student":str(len(student_present)),"student_names":student_present,"missing_student":missing_student})
                else:
                    return json.dumps({"result":"failed"})
#TAKE ATTENDANCE-------------------------------------------------------------------------------------------------------------------


# HAND DETECTION
@app.route('/question')
def question():
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    print("in hand model")
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)






        if results.multi_hand_landmarks:
          image_height, image_width, _ = image.shape
          for hand_landmarks in results.multi_hand_landmarks:
            print(image_width/2)
            half_width=int(image_width/2)
            half_height=int(image_height/2)
            end_point=(half_width,image_height)
            # Blue color in BGR
            color = (255, 0, 0)
            # White
            another_color=(255,255,255)
            # Line thickness of 2 px
            thickness = 2
            start_point=(0,0)
            another_start_point=(half_width,0)
            another_end_point=(image_width,image_height)
            cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(image, another_start_point, another_end_point, another_color, thickness)
            x=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
            y=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_width

            if(half_width<x<image_width):
                print("left")
                return json.dumps({"location":"left"})
            if(0<x<half_width ):
                return json.dumps({"location":"right"})
            print("test"+str(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width))
            print(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # cap.release()


if __name__=="__main__":
    app.run(debug=True)