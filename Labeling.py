import os
import pandas as pd

# Define paths to your data directories
frames_dir = "Test Frames"
with_chd_dir = os.path.join(frames_dir, "With_CHD")
without_chd_dir = os.path.join(frames_dir, "Without_CHD")

# Create lists to store frame file paths and labels
file_paths = []
labels = []

# Helper function to process frames in a given directory
def process_frames(directory, label):
    for video_folder in os.listdir(directory):
        video_folder_path = os.path.join(directory, video_folder)
        if os.path.isdir(video_folder_path):
            for frame in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, frame)
                if os.path.isfile(frame_path):
                    file_paths.append(frame_path)
                    labels.append(label)

# Process frames
process_frames(without_chd_dir, 0)  # Label 0 for without CHD
process_frames(with_chd_dir, 1)     # Label 1 for with CHD

# Create a DataFrame to store the data
data = pd.DataFrame({
    "file_path": file_paths,
    "label": labels
})

# Save the DataFrame to a CSV file
output_csv = "labeled_frames.csv"
data.to_csv(output_csv, index=False)

print(f"Labels saved to {output_csv}")
