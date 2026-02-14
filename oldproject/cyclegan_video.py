import cv2
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import gdown
import argparse
import shutil

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Pix2PixGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(Pix2PixGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                      nn.InstanceNorm2d(dim),
                      nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                      nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def load_cyclegan_generator(model_path):
    netG = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in state_dict:
        state_dict = state_dict['model']
    netG.load_state_dict(state_dict, strict=False)
    netG.eval()
    return netG

def download_model():
    # The model is already downloaded, just return the path
    model_path = 'checkpoints/style_ukiyoe_pretrained/latest_net_G.pth'
    return model_path

class VideoFrameDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames_dir = frames_dir
        self.transform = transform
        self.frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_path = os.path.join(self.frames_dir, self.frame_files[idx])
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.frame_files[idx]

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{frames_dir}/frame_{idx:05d}.png", frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames to {frames_dir}/")
    return idx

def create_video_from_frames(frames_dir, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in the directory")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")

def main():
    # Argument parser for video and style model path
    parser = argparse.ArgumentParser(description='Apply CycleGAN style transfer to a video.')
    parser.add_argument('--video_path', type=str, default='UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03.avi', help='Path to the input video file')
    parser.add_argument('--style_model_path', type=str, default='checkpoints/style_ukiyoe_pretrained/latest_net_G.pth', help='Path to the pretrained style model (.pth)')
    parser.add_argument('--output_video_path', type=str, default='stylized_video.mp4', help='Path to save the stylized output video')
    parser.add_argument('--style_weight', type=float, default=0.5, help='Weight for the style (0=original, 1=full style)')
    parser.add_argument('--output_fps', type=int, default=30, help='Frame rate (fps) for the output video')
    args = parser.parse_args()

    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = f'frames_{video_name}'
    stylized_frames_dir = f'stylized_frames_{video_name}'
    output_video_path = args.output_video_path
    style_model_path = args.style_model_path
    style_weight = args.style_weight
    output_fps = args.output_fps

    # Ensure clean directories
    for d in [frames_dir, stylized_frames_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Extract frames
    num_frames = extract_frames(video_path, frames_dir)

    # Setup transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset and dataloader
    dataset = VideoFrameDataset(frames_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load pix2pix model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_cyclegan_generator(style_model_path)
    model = model.to(device)

    # Process frames
    print(f"Applying style transfer using model: {style_model_path}")
    with torch.no_grad():
        for i, (frame, frame_name) in enumerate(dataloader):
            frame = frame.to(device)
            stylized_frame = model(frame)
            # Denormalize both
            stylized_frame = stylized_frame * 0.5 + 0.5
            orig = frame * 0.5 + 0.5
            # Blend
            blended = style_weight * stylized_frame + (1 - style_weight) * orig
            # Save the blended frame
            output_path = os.path.join(stylized_frames_dir, frame_name[0])
            save_image(blended, output_path)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_frames} frames")
    # Create video from stylized frames
    create_video_from_frames(stylized_frames_dir, output_video_path, fps=output_fps)
    print(f"Stylized video saved to {output_video_path}")

if __name__ == "__main__":
    main() 