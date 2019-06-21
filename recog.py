import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

aayush_image = face_recognition.load_image_file("aayush.jpg")
aayush_face_encoding = face_recognition.face_encodings(aayush_image)[0]

#yash_image = face_recognition.load_image_file("yash.jpg")
#yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

rushikesh_image = face_recognition.load_image_file("rushikesh.jpg")
rushikesh_face_encoding = face_recognition.face_encodings(rushikesh_image)[0]

sir_image = face_recognition.load_image_file("sir.jpg")
sir_face_encoding = face_recognition.face_encodings(sir_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    aayush_face_encoding,
    #yash_face_encoding,
    rushikesh_face_encoding,
    sir_face_encoding,
]
known_face_names = [
    "Aayush",
   # "Yash",
    "Rushikesh",
    "Sir",
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    
	# Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	# Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == 27:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
