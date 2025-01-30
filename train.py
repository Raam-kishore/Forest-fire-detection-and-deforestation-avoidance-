from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO("yolo11m.pt")

    # Check if GPU is available and force it to use the GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the model with optimized settings for your system
    model.train(
        data="data.yaml",       # Path to your data YAML file
        imgsz=640,              # Image size
        batch=18,               # Increase batch size (adjust depending on VRAM usage)
        epochs=100,             # Number of epochs
        device="cuda",          # Specify the device (GPU or CPU)
        workers=1,              # Maximize CPU core usage by increasing the number of workers
  

    )
