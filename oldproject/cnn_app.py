import streamlit as st
import os
from PIL import Image
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
from tqdm import tqdm
import PIL.Image
import subprocess
from pathlib import Path
import shutil

from videoStyle import extract_frames, style_transfer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_img_from_file(file):
    """Load and preprocess an image from a file-like object or file path"""
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file, 'getvalue'):
            img_data = file.getvalue()
        # Handle regular file object
        else:
            img_data = file.read()
        
        # Convert to tensor
        img = tf.image.decode_image(img_data, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        # Resize if needed
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = 512/long_dim
        new_shape = tf.cast(shape*scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise

def validate_image(image_file):
    try:
        img = Image.open(image_file)
        img.verify()  # Verify it's an image
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def validate_video(video_file):
    try:
        # Save to temporary file to check with OpenCV
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_file.getvalue())
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return False, "Could not open video file"
        
        # Check if video has frames
        ret, _ = cap.read()
        if not ret:
            return False, "Video file has no frames"
        
        cap.release()
        os.unlink(tmp_path)  # Clean up temp file
        return True, None
    except Exception as e:
        return False, f"Invalid video file: {str(e)}"

def create_streamlit_app():
    st.set_page_config(
        page_title="Video Style Transfer",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 Video Style Transfer")
    st.markdown("""
    This app allows you to transfer artistic styles from an image to a video.
    Upload a video and a style image, and the app will create a new video with the style applied.
    """)

    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    
    with col2:
        style_file = st.file_uploader("Upload a style image", type=['jpg', 'jpeg', 'png'])
    
    # Parameters
    st.sidebar.header("Parameters")
    frame_freq = st.sidebar.slider("Frame frequency", 1, 60, 30,
                                 help="Extract one frame every N frames")
    fps = st.sidebar.slider("Output video FPS", 1, 30, 8,
                           help="Frames per second in the output video")
    
    if video_file and style_file:
        if st.button("Start Style Transfer"):
            with st.spinner("Processing..."):
                # Create a permanent output directory
                output_dir = "styled_videos"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    video_path = os.path.join(temp_dir, "input_video.mp4")
                    with open(video_path, "wb") as f:
                        f.write(video_file.getvalue())
                    
                    # Load style image
                    style_img = load_img_from_file(style_file)
                    
                    # Create directories for frames
                    frames_dir = os.path.join(temp_dir, "frames")
                    styled_frames_dir = os.path.join(temp_dir, "styled_frames")
                    
                    # Extract frames
                    st.text("Extracting frames from video...")
                    extract_frames(video_path, frames_dir, frame_freq)
                    
                    # Apply style transfer
                    st.text("Applying style transfer to frames...")
                    style_all_frames(frames_dir, styled_frames_dir, style_img)
                    
                    # Create output video
                    st.text("Creating final video...")
                    output_video_path = os.path.join(output_dir, "styled_video.mp4")
                    create_video_from_frames(styled_frames_dir, output_video_path, fps)
                    
                    # Display the result
                    st.success("Style transfer complete!")
                    
                    # Read and display the video
                    with open(output_video_path, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="Download styled video",
                        data=video_bytes,
                        file_name="styled_video.mp4",
                        mime="video/mp4"
                    )

def style_all_frames(original_frame_dir, style_output_dir, style_img):
    """Apply style transfer to all frames"""
    os.makedirs(style_output_dir, exist_ok=True)
    
    frame_files = [f for f in os.listdir(original_frame_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame_file in enumerate(frame_files):
        try:
            input_path = os.path.join(original_frame_dir, frame_file)
            output_path = os.path.join(style_output_dir, frame_file)
            
            # Load and process frame
            with open(input_path, 'rb') as f:
                content_img = load_img_from_file(f)
            
            # Apply style transfer
            result = style_transfer(content_img, style_img)
            result.save(output_path)
            
            # Update progress
            progress = (i + 1) / len(frame_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {i + 1} of {len(frame_files)}")
            
        except Exception as e:
            st.error(f"Error processing frame {frame_file}: {str(e)}")

def create_video_from_frames(frame_dir, output_video_path, fps=8):
    """Create video from styled frames"""
    frame_pattern = os.path.join(frame_dir, "*.jpg")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    create_streamlit_app()

