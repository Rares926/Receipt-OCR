import cv2
import numpy as np
import matplotlib.pyplot as plt

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plot_gray(image):
    plt.figure(figsize=(10,10))
    plt.title('Input Image')
    return plt.imshow(image, cmap='Greys_r')


def contour_to_rect(contour, resize_ratio):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio


def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def draw_text_on_blank_side_by_side(
        image_path,
        ocr_result,
        output_path=None,
        show=True,
        font_scale=0.8,
        font_thickness=1):
    """
    Create a blank image of same size and draw extracted text at the box locations.
    Display original and annotated side by side.

    Args:
        image_path: Path or numpy array of the input image.
        ocr_result: OCR result in PaddleOCR format.
        output_path: Optional path to save side-by-side result.
        show: Whether to show the image.
        font_scale: Font size for text drawing.
        font_thickness: Font thickness for text drawing.
    """

    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path

    h, w, _ = img.shape
    blank_img = np.ones_like(img) * 255  # white image

    text_color = (0, 0, 0)  # black

    for line in ocr_result[0]:
        box = np.array(line[0], dtype=np.int32)
        text = line[1][0]

        # Get bottom-left corner of the text
        x = int(np.min(box[:, 0]))
        y = int(np.min(box[:, 1]))

        # Draw the text on the blank image
        cv2.putText(
            blank_img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    # Concatenate original and text-drawn image side by side
    side_by_side = np.concatenate((img, blank_img), axis=1)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(side_by_side)
        plt.title('Processed Image/OCR Results')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def draw_ocr_results(
        image_path,
        ocr_result,
        output_path=None,
        show=True,
        draw_bboxes=True,
        draw_text=True,
        draw_score=True,
        font_scale=0.5,
        font_thickness=1,
        text_box_alpha=0.0):
    """
    Draw OCR results (boxes and text) on an image

    Args:
        image_path: Path to the input image
        ocr_result: PaddleOCR result (list of boxes and texts)
        output_path: Path to save the output image (if None, won't save)
        show: Whether to display the result using matplotlib
        draw_bboxes: Whether to draw bounding boxes
        draw_text: Whether to draw text
        draw_score: Whether to draw score
        font_scale: Font size of the text
        font_thickness: Thickness of the text
        text_box_alpha: Transparency of text background (0.0 = transparent, 1.0 = opaque)
    Returns:
        The image with boxes and text drawn
    """
    # Read the image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path

    img_draw = img.copy()
    overlay = img.copy()  # for transparency effect

    box_color = (0, 255, 0)        # Green
    text_color = (255, 0, 0)       # Red (RGB)
    text_bg_color = (128, 128, 128)  # Gray background for text box

    for line in ocr_result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]

        box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))

        if draw_bboxes:
            cv2.polylines(img_draw, [box], True, box_color, 2)

        text_pos = (min(box[:, 0, 0]), min(box[:, 0, 1]) - 10)
        display_text = f"{text} ({confidence:.2f})" if draw_score else text

        if draw_text:
            # Get size of text for background
            (text_width, text_height), _ = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            top_left = (text_pos[0], text_pos[1] - text_height - 4)
            bottom_right = (text_pos[0] + text_width, text_pos[1] + 4)

            # Draw semi-transparent background
            if text_box_alpha > 0:
                cv2.rectangle(overlay, top_left, bottom_right, text_bg_color, -1)
                cv2.addWeighted(overlay, text_box_alpha, img_draw, 1 - text_box_alpha, 0, img_draw)

            # Draw the text itself
            cv2.putText(
                img_draw,
                display_text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness
            )

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx