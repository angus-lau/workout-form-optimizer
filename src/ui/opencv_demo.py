import cv2 as cv


def opencv() -> None:
    """
    Opens the first working webcam (cycles from 0-4) and displays its feed
    Press 'q' to quit.
    """
    for i in range(5):
        # cycles through the possible camera input options
        capture = cv.VideoCapture(i)

        if not capture.isOpened():
            capture.release()
            continue

        while True:
            is_true, frame = capture.read()
            if not is_true:
                print("Failed to read frame")
                break

            cv.imshow("Webcam", frame)

            # press q to exit the webcam and release all the capture frames
            if cv.waitKey(20) & 0xFF == ord("q"):
                break
        capture.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
