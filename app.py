import cv2
import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def mark_largest_moving_object(frame, bg_subtractor, model, class_names, object_name="Bird"):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological transformations to reduce noise
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the moving object)
    largest_contour = max(contours, key=cv2.contourArea, default=None)

    # Draw rectangle around the largest moving object and display the name
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and preprocess the image for prediction
        object_image = frame[y:y+h, x:x+w]
        object_image = cv2.resize(object_image, (224, 224))  # Assuming model input size is 224x224
        object_image = image.img_to_array(object_image)
        object_image = np.expand_dims(object_image, axis=0)
        object_image = object_image / 255.0  # Normalize pixel values

        # Make predictions using the model
        predictions = model.predict(object_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # Display the predicted class name inside the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(predicted_class_name, font, font_scale, font_thickness)[0]
        text_position = (int(x + (w - text_size[0]) / 2), int(y + h + text_size[1] + 5))
        cv2.putText(frame, predicted_class_name, text_position, font, font_scale, (255, 255, 255), font_thickness)

    return frame

def extract_images_and_display(video_path, output_folder, model, class_names, max_images=100):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter for saving frames
    output_path = os.path.join(output_folder, "result.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create folder to store extracted images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_count = 0

    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Mark the largest moving object on the frame and predict its class
        marked_frame = mark_largest_moving_object(frame.copy(), bg_subtractor, model, class_names, object_name="Bird")

        # Save the original frame without marking
        image_path = os.path.join(output_folder, f"extracted_image_{image_count + 1}.png")
        cv2.imwrite(image_path, frame)

        # Display the video with the largest moving object outlined and predicted class
        cv2.imshow('Video with Largest Moving Object Outlined', marked_frame)

        # Write the frame to the output video
        out.write(marked_frame)

        # Increment image count
        image_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the trained model
    model = load_model("model_inception.h5")

    # Set the class names used during training
    class_names = {
        0: 'ABBOTTS BABBLER', 1: 'ABBOTTS BOOBY', 2: 'ABYSSINIAN GROUND HORNBILL', 3: 'AFRICAN CROWNED CRANE',
        4: 'AFRICAN EMERALD CUCKOO', 5: 'AFRICAN FIREFINCH', 6: 'AFRICAN OYSTER CATCHER', 7: 'AFRICAN PIED HORNBILL',
        8: 'AFRICAN PYGMY GOOSE', 9: 'ALBATROSS', 10: 'ALBERTS TOWHEE', 11: 'ALEXANDRINE PARAKEET', 12: 'ALPINE CHOUGH',
        13: 'ALTAMIRA YELLOWTHROAT', 14: 'AMERICAN AVOCET', 15: 'AMERICAN BITTERN', 16: 'AMERICAN COOT', 17: 'AMERICAN FLAMINGO',
        18: 'AMERICAN GOLDFINCH', 19: 'AMERICAN KESTREL'
    }

    # Set the path to the video file
    video_path = "bird.mp4"

    # Set the output folder to store extracted images
    output_folder = "outputImage/images"

    # Extract up to 25 images from the video, predict the class, and store in the output folder
    extract_images_and_display(video_path, output_folder, model, class_names, max_images=100)
