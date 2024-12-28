import cv2
import numpy as np
import os
print('hello world')
def normalize_color(frame, normalize_type='rescale', convert_to_grayscale=False):
    """
    Normalize the color of the frame (either rescale or standardize).
    Optionally convert to grayscale.
    """
    if convert_to_grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    if normalize_type == 'rescale':
        frame = frame / 255.0  # Normalize to the range [0, 1]
    elif normalize_type == 'standardize':
        frame = (frame - np.mean(frame)) / (np.std(frame) + 1e-7)  # Standardize the frame

    return frame

def normalize_frames_in_folder(input_folder, output_folder, normalize_type='rescale', convert_to_grayscale=False):
    """
    Normalize frames in the given folder and save them to the output folder.
    Optionally convert frames to grayscale.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    frame_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]  # Get all .jpg files

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)  # Read the frame image
        normalized_frame = normalize_color(frame, normalize_type, convert_to_grayscale)  # Normalize the frame

        # Save the normalized frame to the output folder
        output_frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_frame_path, normalized_frame * 255)  # Save it back in [0, 255] range if rescaled
        print(f"Normalized and saved frame: {frame_file}")
