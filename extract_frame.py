import os
import sys

import cv2


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_frame.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_num = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)

        # Wait for the user to press a key
        key = cv2.waitKey(33)

        # Save the frame if the user pressed the enter key
        if key == 13:
            output_path = os.path.join(
                output_dir,
                f"{os.path.basename(video_file).rstrip('.mp4')}_{frame_num}.jpg",
            )
            cv2.imwrite(output_path, frame)
            print("Frame saved")

        # Exit if the user pressed the escape key
        elif key == 27:
            break

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
