# extracting.py
import cv2
import os
import argparse

def extract_frames_from_video(video_path, output_folder, frame_rate=30, resize_dim=(224, 224)):
    """
    Extract frames from a video file and save them as images in the specified folder.
    """
    if not os.path.isfile(video_path):
        print(f"Error: {video_path} is not a valid file.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    print(f"Processing video: {video_path}, FPS: {fps}, Total frames: {total_frames}")

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    extracted_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        if frame_count % max(1, int(fps / frame_rate)) == 0:  # Ensure valid divisor
            resized_frame = cv2.resize(frame, resize_dim)

            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            extracted_frames.append(resized_frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from {video_path}.")
    return extracted_frames


def extract_frames_from_multiple_videos(video_folder, output_folder, frame_rate=30, resize_dim=(224, 224)):
    """
    Process multiple videos in a folder, extracting frames from each and saving them in separate subfolders.
    """
    if not os.path.isdir(video_folder):
        print(f"Error: {video_folder} is not a valid directory.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        print(f"No video files found in {video_folder}.")
        return

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_subfolder = os.path.join(output_folder, os.path.splitext(video_file)[0])
        os.makedirs(output_subfolder, exist_ok=True)

        extract_frames_from_video(video_path, output_subfolder, frame_rate, resize_dim)

    print(f"Processed {len(video_files)} videos. Frames saved in {output_folder}.")


# The following block allows the script to be run both directly or imported into other scripts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("video_folder", help="Path to the folder containing video files.")
    parser.add_argument("output_folder", help="Path to the folder where extracted frames will be saved.")
    parser.add_argument("--frame_rate", type=int, default=30, help="Number of frames to extract per second.")
    parser.add_argument("--resize_dim", type=int, nargs=2, default=(224, 224), help="Resize dimensions (height, width).")

    args = parser.parse_args()

    extract_frames_from_multiple_videos(args.video_folder, args.output_folder, args.frame_rate, tuple(args.resize_dim))
