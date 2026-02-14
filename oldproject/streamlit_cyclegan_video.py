import streamlit as st
import os
import subprocess
import tempfile
import shutil

# Define available styles and their model paths
STYLE_MODELS = {
    'Monet': 'checkpoints/style_monet_pretrained/latest_net_G.pth',
    'Ukiyo-e': 'checkpoints/style_ukiyoe_pretrained/latest_net_G.pth',
    'Cezanne': 'checkpoints/style_cezanne_pretrained/latest_net_G.pth',
    'Van Gogh': 'checkpoints/style_vangogh_pretrained/latest_net_G.pth',
    'Photo2Monet': 'checkpoints/monet2photo_pretrained/latest_net_G.pth',
    # Add more styles as needed
}

st.title('CycleGAN Video Style Transfer')
st.write('Upload a video and choose a style to apply!')

uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov'])
style = st.selectbox('Choose a style', list(STYLE_MODELS.keys()))
style_weight = st.slider('Style weight', 0.0, 1.0, 0.5, 0.01)

if uploaded_file and style:
    if st.button('Apply Style Transfer'):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded video to temp file
            input_video_path = os.path.join(tmpdir, uploaded_file.name)
            with open(input_video_path, 'wb') as f:
                f.write(uploaded_file.read())
            # Output video path
            base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            output_video_path = os.path.join(tmpdir, f'{base_name}_{style}.mp4')
            # Build command
            command = [
                'python', 'cyclegan_video.py',
                '--video_path', input_video_path,
                '--style_model_path', STYLE_MODELS[style],
                '--output_video_path', output_video_path,
                '--style_weight', str(style_weight)
            ]
            st.info('Processing video, this may take a while...')
            with st.spinner('Applying style transfer...'):
                result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_video_path):
                st.success('Style transfer complete!')
                with open(output_video_path, 'rb') as f:
                    st.download_button('Download stylized video', f, file_name=f'{base_name}_{style}.mp4')
            else:
                st.error('There was an error processing your video.')
                st.text(result.stdout)
                st.text(result.stderr) 