import cv2
import os

def convert_all_frames_to_grayscale(parent_folder, output_parent_folder):
    """
    Converts all images in multiple subfolders to grayscale and saves them in corresponding subfolders in the output folder.
    
    :param parent_folder: Parent folder containing subfolders of frames.
    :param output_parent_folder: Parent folder where grayscale images will be saved, mirroring the input structure.
    """
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Full path of the input image
                input_path = os.path.join(root, file)
                
                # Create corresponding output path
                relative_path = os.path.relpath(root, parent_folder)
                output_dir = os.path.join(output_parent_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)
                
                # Read the image
                image = cv2.imread(input_path)
                
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Save the grayscale image
                cv2.imwrite(output_path, gray_image)
                print(f"Converted {input_path} to grayscale and saved to {output_path}")

# Example usage
parent_folder = "Test_Frames\Without_CHDs"  # Replace with the path to your main folder containing subfolders
output_parent_folder = "Test_Frames_Grayscale\Without_CHDs"  # Replace with the path where grayscale images will be saved

convert_all_frames_to_grayscale(parent_folder, output_parent_folder)
