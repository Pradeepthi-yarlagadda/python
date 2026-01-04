import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#load know faces

student1_image = face_recognition.load_image_file("faces/student1.jpg")
student1_encoding = face_recognition.face_encodings(student1_image)[0]

student2_image = face_recognition.load_image_file("faces/student2.jpg")
student2_encoding = face_recognition.face_encodings(student2_image)[0]

student3_image = face_recognition.load_image_file("faces/student3.jpg")
student3_encoding = face_recognition.face_encodings(student3_image)[0]

student4_image = face_recognition.load_image_file("faces/student4.jpg")
student4_encoding = face_recognition.face_encodings(student4_image)[0]

known_face_encodings=[student1_encoding,student2_encoding,student3_encoding,student4_encoding]
known_face_names=["student1","student2","student3","student4"]

#list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

#get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    small_frame = cv2.resize(frame, (0,0), fx=0.25 ,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recognize face
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        name="unknown"
        if(matches[best_match_index]):
            name = known_face_names[best_match_index]

        #add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,200)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " present", bottomLeftCornerOfText , font , fontScale , fontColor, thickness , lineType)

        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])
    
    cv2.imshow("Attendence",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()