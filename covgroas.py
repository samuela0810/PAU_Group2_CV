# main_script.py
from __future__ import print_function


import importlib
import subprocess
import sys

# Auto-install packages from requirements_script.py
def install_missing_packages():
    try:
        from cv_requirements_script import required_packages
    except ImportError:
        print("Missing cv_requirements_script.py!")
        sys.exit(1)

    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_missing_packages()

# ---------------- MAIN PROGRAM ---------------- #

from builtins import input
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse

# Histogram plotting function
def show_histograms(original, modified):
    color = ('b', 'g', 'r')
    plt.figure(figsize=(12, 6))

    # Histogram of original image
    plt.subplot(1, 2, 1)
    for i, col in enumerate(color):
        hist = cv.calcHist([original], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram - Original Image')

    # Histogram of modified image
    plt.subplot(1, 2, 2)
    for i, col in enumerate(color):
        hist = cv.calcHist([modified], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram - Modified Image')

    plt.show()

# Argument parser for image input
parser = argparse.ArgumentParser(description='Change contrast and brightness of an image, and show histograms.')
parser.add_argument('--input', help='Path to input image.', required=True)
args = parser.parse_args()

# Load image using OpenCV
image = cv.imread(cv.samples.findFile(args.input))
if image is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Create an empty image for output
new_image = np.zeros(image.shape, image.dtype)

# Get alpha and beta values from user
print(' Basic Linear Transforms ')
print('-------------------------')
try:
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
except ValueError:
    print('Error, not a number')
    exit(0)

# Apply linear transform manually
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

# Show original and modified images
cv.imshow('Original Image', image)
cv.imshow('Modified Image', new_image)

# Show histograms
show_histograms(image, new_image)

# Wait for key press
cv.waitKey(0)
cv.destroyAllWindows()

# To run the code, type "python covgroas.py --input the_path_of_the_image_you_want_to_modify.jpg(jpeg,png,...)" in the terminal
