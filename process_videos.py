import os

import cv2

from utils import process_image


def main():
    """Process all the videos in the `videos` directory and save them to the output_video directory"""
    video_dir = "videos"
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        output_path = os.path.join(output_dir, video_name)

        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing {video_name}...", end="", flush=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_image(frame)

            out.write(frame)

        cap.release()
        out.release()

        print("Done")


if __name__ == "__main__":
    main()
