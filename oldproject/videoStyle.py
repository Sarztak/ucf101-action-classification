import cv2
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img,channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1],tf.float32)
    long_dim = max(shape)
    scale = max_dim/long_dim
    new_shape = tf.cast(shape*scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def style_transfer(content_img, style_img):
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    styled_img = hub_model(tf.constant(content_img), tf.constant(style_img))[0]

    return tensor_to_image(styled_img)

def extract_frames(video_path, output_dir, frame_freq=30):
    """
    Extract frames from a video file
    Args:
        video_path: path to video file
        output_dir: directory to save frames
        frame_freq: extract one frame every frame_freq frames
    """
    print(f"\nStarting extract_frames for video: {video_path}")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    style_output_dir = os.path.join(os.path.dirname(output_dir), "test_style")
    os.makedirs(style_output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return 0
    
    print(f"Successfully opened video: {video_path}")
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_freq == 0:
            # Generate output filename
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{saved_count:03d}.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            style_output_path = os.path.join(style_output_dir, frame_filename)
            
            # Save the original frame
            cv2.imwrite(output_path, frame)
            print(f"Saved original frame to: {output_path}")
            
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Total frames saved: {saved_count}")
    return saved_count

def style_all_frames(original_frame_dir, style_output_dir):
    """
    Apply style transfer to all frames in a directory
    Args:
        original_frame_dir: directory containing original frames
        style_output_dir: directory to save styled frames
    """
    os.makedirs(style_output_dir, exist_ok=True)
    
    # Get all image files in the directory
    frame_files = [f for f in os.listdir(original_frame_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for frame_file in tqdm(frame_files, desc="Styling frames"):
        try:
            # Construct full paths
            input_path = os.path.join(original_frame_dir, frame_file)
            output_path = os.path.join(style_output_dir, frame_file)
            
            # Grab frame image
            content_img = load_img(input_path)
            print(f"Loaded content image: {frame_file}")
            result = style_transfer(content_img, STYLE_IMG)
            print("Applied style transfer")
            
            # Save the style-transferred frame
            result.save(output_path)
            print(f"Saved style-transferred frame to: {output_path}")
            
        except Exception as e:
            print(f"Error during style transfer for {frame_file}: {str(e)}")

def process_all_videos(input_base_dir, output_base_dir, frame_freq=30):
    """
    Process all videos in the input directory and its subdirectories
    Args:
        input_base_dir: base directory containing class folders with videos
        output_base_dir: base directory where frames will be saved
        frame_freq: extract one frame every frame_freq frames
    """
    print("\nStarting process_all_videos function...")
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_base_dir) 
                 if os.path.isdir(os.path.join(input_base_dir, d)) and not d.startswith('.')]
    
    print(f"Found class directories: {class_dirs}")
    total_frames = 0
    
    # Process each class directory
    for class_name in class_dirs:
        print(f"\nProcessing class: {class_name}")
        
        input_class_dir = os.path.join(input_base_dir, class_name)
        output_class_dir = os.path.join(output_base_dir, class_name)
        
        # Get all video files in the class directory
        video_files = [f for f in os.listdir(input_class_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Found {len(video_files)} video files in {class_name}")
        
        # Process each video file with progress bar
        for video_file in tqdm(video_files, desc=f"Processing videos in {class_name}"):
            video_path = os.path.join(input_class_dir, video_file)
            print(f"\nProcessing video: {video_path}")
            frames_saved = extract_frames(video_path, output_class_dir, frame_freq)
            total_frames += frames_saved
    
    print(f"\nTotal frames extracted: {total_frames}")

if __name__ == "__main__":
    # Set matplotlib to use MacOSX backend
    import matplotlib
    matplotlib.use('MacOSX')
    
    # Test with a single video file
    video_path = "data/misc/Bowen's Straight - SNL.mp4"
    output_dir = "data/sampled_frames/test"
    frame_freq = 30

    print("Starting script...")
    STYLE_IMG = load_img("/Users/blag/Documents/UChicago MS/2025 Spring/Computer Vision with DL/inPainting/daliMemory.jpg")
    print("Loaded style image")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, video_path)
    output_dir = os.path.join(current_dir, output_dir)
    
    print(f"Processing video: {video_path}")
    print(f"Saving frames to: {output_dir}")
    print(f"Frame frequency: {frame_freq}")
    
    # Process single video
    frames_saved = extract_frames(video_path, output_dir, frame_freq)
    print(f"Total frames saved: {frames_saved}")

    # Process styling frames
    style_all_frames(output_dir, 'data/sampled_frames/Bowen_dali')
    print(f"Finished styling frames")