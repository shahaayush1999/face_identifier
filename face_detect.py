import cv2
from time import time
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frames = []

while(True):
	time_start = time()
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)


	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
		break #comment break to allow more rectangles

	time_end = time()

	# Display the resulting frame and stats
	print("Found {number} faces at {fps} fps".format(number = len(faces), fps = 1 / (time_end - time_start)))
	cv2.imshow('frame', frame)
	
	# Exit if ESC pressed
	if cv2.waitKey(1) == 27:
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
