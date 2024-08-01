import torch
from torch.utils.data import Dataset
import os
import csv
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = os.path.abspath(root_dir)  # Get absolute path
        self.transform = transform
        self.video_labels = self._load_csv(csv_file)
        self.classes = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]
        self.video_files = list(self.video_labels.keys())
        
    def _load_csv(self, csv_file):
        video_labels = {}
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                filename, label = row
                full_path = os.path.join(self.root_dir, filename)
                if os.path.exists(full_path):
                    video_labels[filename] = label
                else:
                    print(f"Warning: File not found - {full_path}")
                    print(f"Current working directory: {os.getcwd()}")
                    print(f"Contents of {self.root_dir}:")
                    for file in os.listdir(self.root_dir):
                        print(file)
        
        if not video_labels:
            raise ValueError(f"No valid video files found in {self.root_dir}")
        
        return video_labels

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.video_labels[video_file]
        video_path = os.path.join(self.root_dir, video_file)
        
        # Load video
        try:
            video, _, _ = read_video(video_path, pts_unit='sec')
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a dummy tensor and label if video can't be loaded
            return torch.zeros((3, 16, 224, 224)), self.classes.index("Not-Engaged")
        
        # Sample frames (every 5th frame)
        video = video[::5]
        
        # Ensure we have a fixed number of frames (16)
        if video.shape[0] < 16:
            video = video.repeat(16 // video.shape[0] + 1, 1, 1, 1)[:16]
        elif video.shape[0] > 16:
            video = video[:16]
        
        # Apply transforms
        if self.transform:
            video = self.transform(video)
        
        # Convert label to numeric
        label_idx = self.classes.index(label)
        
        return video, label_idx

    def get_video_files(self):
        return self.video_files