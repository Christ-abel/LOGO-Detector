import os

# Fix for KMP duplicate libs (needed for some environments like Anaconda)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from roboflow import Roboflow
from ultralytics import YOLO

# Load base YOLOv8 model for new training

model = YOLO("yolov8n.pt")

def download_dataset():
    """Optional: download dataset from Roboflow (only if needed)."""
    rf = Roboflow(api_key="kj8aqDdJvUbEYv9LJbfa")
    project = rf.workspace("ngochoang").project("logo-lvjnq")
    dataset = project.version(3).download("yolov8")
    print("Dataset downloaded.")

def train_yolov8():
    """Train the model on your dataset."""
    model.train(data="c:/Users/HP/Downloads/Logo.v3i.yolov8/data.yaml", epochs=10, imgsz=640)
    print("Training complete. Check the 'runs/detect/train' folder for results.")

def predict_image(image_path):
    """Predict a single image using the best trained model."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Load the newly trained model
    trained_model = YOLO("C:/Users/HP/OneDrive/Desktop/LOGO3/runs/detect/train2/weights/best.pt")
    results = trained_model.predict(source=image_path)
    for r in results:
        try:
            r.show()
        except:
            print("GUI not supported. Skipping image display.")
    print("Image prediction complete.")
    return results

def predict_webcam():
    """Predict from webcam using the best trained model."""
    trained_model = YOLO("runs/detect/train/weights/best.pt")
    trained_model.predict(source=0, save=True, project="output_folder", name="webcam_run")
    print("Webcam prediction complete. Results saved in 'output_folder/webcam_run'.")

def main():
    # Optional: download the dataset if not done already
    if input("Download dataset? (yes/no): ").strip().lower() == "yes":
        download_dataset()

    # Ask to train the model
    if input("Train the model now? (yes/no): ").strip().lower() == "yes":
        train_yolov8()

    # Ask to predict using webcam or image
    choice = input("Type 'webcam' to use webcam or paste the full path to an image file: ").strip()
    if choice.lower() == "webcam":
        predict_webcam()
    else:
        predict_image(choice)

if __name__ == "__main__":
    main()
