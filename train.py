import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize, Normalize
import torch.nn as nn
from transformers import TrainingArguments, Trainer

import os
from dataset import VideoDataset
from model import ViViTForClassification
from utils import compute_metrics
from config import Config

def train_vivit_model():
    config = Config()

    print(f"Video directory: {os.path.abspath(config.VIDEO_DIR)}")
    print(f"CSV file: {os.path.abspath(config.CSV_FILE)}")

    # Define transforms
    transform = nn.Sequential(
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    # Create dataset
    dataset = VideoDataset(root_dir=config.VIDEO_DIR, 
                           csv_file=config.CSV_FILE, 
                           transform=transform)

    # Print dataset info for debugging
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First item in dataset: {dataset[0][0].shape}, {dataset[0][1]}")
    
    # Print the first 10 video files being used
    print("First 10 video files:")
    for file in dataset.get_video_files()[:10]:
        print(file)

    # Split into train and validation sets
    total_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create model
    model = ViViTForClassification(num_classes=config.NUM_CLASSES)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOG_DIR,
        logging_steps=config.LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Save the model
    trainer.save_model(config.SAVE_DIR)

if __name__ == "__main__":
    train_vivit_model()