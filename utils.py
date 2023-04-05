import cv2
import numpy as np


def color_selection(image):
    """Apply color selection to the image."""
    # Convert colorspace to HSL
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Mask for white lane lines.
    white_lowerb = np.array([0, 200, 0], dtype=np.uint8)
    white_upperb = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsl_image, white_lowerb, white_upperb)

    # Mask for yellow lane lines.
    yellow_lowerb = np.array([10, 0, 100], dtype=np.uint8)
    yellow_upperb = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsl_image, yellow_lowerb, yellow_upperb)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    """Apply Canny edge detection to the image."""
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """Apply image mask to reduce noises from useless information"""
    mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    """Find line segments using Hough transform."""
    return cv2.HoughLinesP(
        image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )


def draw_lines(image, lines, color=(0, 0, 255), thickness=10):
    """Draw line segments onto the image"""
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def weighted_img(image_1, alpha, image_2, beta, gamma):
    """Return weighted sum of two images."""
    return cv2.addWeighted(image_1, alpha, image_2, beta, gamma)


def split_left_right_lines(lines):
    """Split the lines into left and right lane lines."""
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lane_lines.append(line)
            else:
                right_lane_lines.append(line)

    return left_lane_lines, right_lane_lines


def calculate_average_slope(lines):
    """Calculate the average slope of left and right lane lines."""
    slope = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope += (y2 - y1) / (x2 - x1)

    return slope / len(lines)


def extrapolate_lines(lines, y_min, y_max):
    """Extrapolate the line segments to get the full extent of the lane lines"""
    left_lines, right_lines = split_left_right_lines(lines)
    left_lane_line_slope = calculate_average_slope(left_lines)
    right_lane_line_slope = calculate_average_slope(right_lines)
    left_lines_mean = np.mean(left_lines, axis=0)
    right_lines_mean = np.mean(right_lines, axis=0)

    x1_left = int(
        (y_min - left_lines_mean[0][1]) / left_lane_line_slope + left_lines_mean[0][0]
    )
    x2_left = int(
        (y_max - left_lines_mean[0][1]) / left_lane_line_slope + left_lines_mean[0][0]
    )
    x1_right = int(
        (y_min - right_lines_mean[0][1]) / right_lane_line_slope
        + right_lines_mean[0][0]
    )
    x2_right = int(
        (y_max - right_lines_mean[0][1]) / right_lane_line_slope
        + right_lines_mean[0][0]
    )

    return np.array(
        [[[x1_left, y_min, x2_left, y_max], [x1_right, y_min, x2_right, y_max]]],
        dtype=np.int32,
    )


def process_image(image):
    """Detect lane lines in the image and return the image with lane lines drawn on it."""
    color_selected = color_selection(image)

    gray = grayscale(color_selected)
    blurred_gray = gaussian_blur(gray, 5)
    edges = canny(blurred_gray, 50, 150)

    h, w = image.shape[:2]
    lower_left = [w / 9, h]
    lower_right = [w * 8 / 9, h]
    top_left = [w / 2 - w / 8, h / 2 + h / 10]
    top_right = [w / 2 + w / 8, h / 2 + h / 10]
    vertices = np.array(
        [[lower_left, top_left, top_right, lower_right]],
        dtype=np.int32,
    )
    edges = region_of_interest(edges, vertices)

    lines = hough_lines(edges, 1, np.pi / 180.0, 30, 20, 200)

    lane_lines = extrapolate_lines(lines, h / 2 + h / 10, h)

    line_image = np.zeros_like(image)
    draw_lines(line_image, lane_lines)

    return weighted_img(image, 0.8, line_image, 1.0, 0.0)
