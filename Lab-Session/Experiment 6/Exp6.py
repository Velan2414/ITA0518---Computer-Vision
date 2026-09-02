import cv2
import os
from google.colab import files
from IPython.display import HTML
from base64 import b64encode

# -------------------------------------------------------------
# Step 1: Upload video file
# -------------------------------------------------------------
print("Please upload a video file:")
uploaded = files.upload()

video_path = list(uploaded.keys())[0]

# -------------------------------------------------------------
# Step 2: Function to process video speed
# -------------------------------------------------------------
def process_video_speed(input_path, raw_output_path, speed_factor):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Fallback FPS
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    if speed_factor > 1.0:
        # Fast Motion: Skip frames
        step = int(round(speed_factor))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                out.write(frame)
            frame_count += 1
            
    elif speed_factor < 1.0:
        # Slow Motion: Duplicate frames
        repeat_count = int(round(1.0 / speed_factor))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for _ in range(repeat_count):
                out.write(frame)

    cap.release()
    out.release()

# -------------------------------------------------------------
# Step 3: Process videos and re-encode to H.264 for HTML5
# -------------------------------------------------------------
# Generate raw OpenCV videos
process_video_speed(video_path, "raw_fast.mp4", speed_factor=2.0)
process_video_speed(video_path, "raw_slow.mp4", speed_factor=0.5)

# Convert to web-compatible H.264 format using FFmpeg
os.system("ffmpeg -y -i raw_slow.mp4 -vcodec libx264 slow_motion.mp4 -loglevel quiet")
os.system("ffmpeg -y -i raw_fast.mp4 -vcodec libx264 fast_motion.mp4 -loglevel quiet")

print("Processing complete! Displaying videos below...\n")

# -------------------------------------------------------------
# Step 4: Display Videos inside Google Colab
# -------------------------------------------------------------
def display_video(file_path, title="Video"):
    mp4 = open(file_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'''
    <h3>{title}</h3>
    <video width=500 controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    ''')

# Show Slow Motion Video
display_video("slow_motion.mp4", "1. Slow Motion Video (0.5x)")
