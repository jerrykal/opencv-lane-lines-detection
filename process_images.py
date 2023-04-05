import os

import cv2

from utils import process_image


def main():
    """Process all images in the images directory and save them to the output_images directory."""
    image_dir = os.path.join(os.path.dirname(__file__), "images")
    output_dir = os.path.join(os.path.dirname(__file__), "output_images")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        output_image = process_image(image)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, output_image)


if __name__ == "__main__":
    main()
