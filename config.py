class Config:
    # Data paths
    VIDEO_DIR = r'C:\Users\Rishikesh\BITS\code\SOP\emotion detection\Data' 
    CSV_FILE = r'C:\Users\Rishikesh\BITS\code\SOP\emotion detection\test1.csv' 
    
    # Output paths
    OUTPUT_DIR = './results'
    LOG_DIR = './logs'
    SAVE_DIR = './saved_model'
    
    # Model and training parameters
    NUM_CLASSES = 4
    TRAIN_SPLIT = 0.8
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 10