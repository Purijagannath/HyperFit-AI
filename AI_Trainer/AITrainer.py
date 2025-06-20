import cv2 as cv
import PoseEstimationModule as pm
import os

# Get full path to video file
video_path = os.path.join(os.path.dirname(__file__), "0.mp4")

# Load video
capture = cv.VideoCapture(video_path)

# Check if the video opened correctly
if not capture.isOpened():
    print("Error: Could not load 0.mp4")
    exit()

# Create PoseDetector instance
detector = pm.PoseDetector()

while True:
    success, img = capture.read()
    if not success:
        print("Finished processing video or error reading frame.")
        break

    # Detect pose
    img = detector.findPose(img, draw=True)

    # Get landmark positions
    lmList = detector.getPosition(img, draw=False)

    # If landmarks are detected
    if lmList:
        # Calculate angle at the right arm: shoulder (11), elbow (13), wrist (15)
        angle = detector.getAngle(img, 11, 13, 15, draw=True)
        print(f"Angle at elbow (Right arm): {angle:.2f}°")

    # Display the result
    cv.imshow("Pose Detection - Video", img)

    # Break loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv.destroyAllWindows()
