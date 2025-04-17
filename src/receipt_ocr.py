import cv2
from paddleocr import PaddleOCR
import argparse
import os
import sys
import datetime
from utils import (
    plot_gray, opencv_resize, get_receipt_contour,
    contour_to_rect, wrap_perspective, plot_rgb,
    draw_text_on_blank_side_by_side
)


def preprocess_image(image, target_height=500):
    """Preprocess the input image for receipt detection."""
    # Downscale image as finding receipt contour is more efficient on a small image
    resize_ratio = target_height / image.shape[0]
    original = image.copy()
    resized = opencv_resize(image, resize_ratio)
    
    # Convert to grayscale for further processing
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return original, resized, resize_ratio, blurred


def detect_receipt_edges(blurred_image):
    """Detect edges in the preprocessed image."""
    # Detect white regions
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred_image, rect_kernel)
    
    # Find edges using Canny algorithm
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    
    return edged


def find_receipt_contour(edged_image, original_image):
    """Find and extract the receipt contour from the image."""
    # Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(edged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get 10 largest contours
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Get the receipt contour
    receipt_contour = get_receipt_contour(largest_contours)
    
    return receipt_contour


def perform_ocr(image):
    """Perform OCR on the given image using PaddleOCR."""
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=True
    )
    
    result = ocr.ocr(image, cls=True)
    return result


def display_results(original_image, processed_image, ocr_result, show=True):
    """Display the original and processed images with OCR results."""
    if not show:
        return
        
    # Display the processed receipt
    plot_gray(original_image)
    
    # Draw OCR results on the image
    draw_text_on_blank_side_by_side(
        image_path=processed_image,
        ocr_result=ocr_result,
        show=True
    )

def main():
    """Main function to process a receipt image and extract text."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract text from receipt images using OCR')
    parser.add_argument('-img', '--image_path', type=str, help='Path to the receipt image file')
    parser.add_argument('-v', '--view', action='store_true', help='Display images and visualization of results')
    parser.add_argument('-pp', '--pre_process', action='store_true', help='Pre-process the image')
    parser.add_argument('-log', '--log_path', type=str,default='./logs', help='Path to folder where logs will be saved')

    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: File {args.image_path} does not exist")
        sys.exit(1)
    
    # Load image
    image = cv2.imread(args.image_path)
    
    # Check if image loaded successfully
    if image is None:
        print(f"Error: Could not load image {args.image_path}")
        sys.exit(1)
    
    # Preprocess image
    if args.pre_process:
        original, resized, resize_ratio, blurred = preprocess_image(image)

        # Detect edges
        edged = detect_receipt_edges(blurred)
        
        # Find receipt contour
        receipt_contour = find_receipt_contour(edged, resized)
        
        # Transform perspective to get a top-down view of the receipt
        scanned = wrap_perspective(
            original.copy(), 
            contour_to_rect(receipt_contour, resize_ratio)
        )
    else:
        scanned = image
    
    # Perform OCR
    result = perform_ocr(scanned)
    
    # Print recognized text
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    
    # Display results only if view is enabled
    display_results(image, scanned, result, show=args.view)
    
    # Print first result
    print(result[0])

    # Save log file with timestamp
    if args.log_path:

        os.makedirs(args.log_path, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(args.log_path, f"log_{timestamp}.txt")
        
        # Write log file with input image path and OCR results
        with open(log_filename, 'w') as f:
            f.write(f"Input image --> {args.image_path}\n")
            f.write(f"OCR result --> {str(result[0])}")



if __name__ == "__main__":
    main()
