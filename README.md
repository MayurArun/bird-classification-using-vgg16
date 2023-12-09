# Bird Recognition and Object Tracking

## Overview

This Python script processes a video, identifies the largest moving object (assumed to be a bird), and predicts its class using a pre-trained neural network model. The script employs computer vision techniques with OpenCV, deep learning with Keras, and numerical operations with NumPy.

## Technical Stack

- **OpenCV (`cv2`):** Video processing, image processing, and object detection.
- **Keras (`from keras.*`):** Loading pre-trained models, image preprocessing, and making predictions.
- **NumPy (`numpy`):** Numerical operations and array manipulation.

## Functionality

### mark_largest_moving_object

- Takes a video frame, background subtractor, pre-trained model, class names, and an object name.
- Applies background subtraction, morphological transformations, and contour analysis.
- Identifies and marks the largest moving object on the frame.
- Predicts the class of the object using the pre-trained model.
- Displays the video frame with the marked object and predicted class.

### extract_images_and_display

- Takes a video file path, output folder, pre-trained model, class names, and the maximum number of images.
- Opens the video file, creates a background subtractor, and captures frames.
- Marks the largest moving object, predicts its class, and saves the original frame without marking.
- Displays the video with the largest moving object outlined and the predicted class.
- Writes the marked frames to an output video file.
- Stops processing when the maximum specified number of images is reached or if 'q' is pressed.
- Releases video capture and writer objects.

## How to Implement

1. Install dependencies: `cv2`, `keras`, `numpy`.
2. Download the pre-trained Inception model (`model_inception.h5`) and place it in the script's directory.
3. Set the correct video file path and output folder in the main block of the script.
4. Run the script (`python script_name.py`), and it will process the video, mark the largest moving object, predict its class, and save the results.

**Note:**

- The script assumes the Inception model architecture with an input size of 224x224 pixels.
- Modify the class names dictionary according to the classes used during training.
- Press 'q' in the video window to stop processing and close the window.
