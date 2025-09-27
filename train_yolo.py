from ultralytics import YOLO
from roboflow import Roboflow

#Download Your Dataset From Roboflow

rf = Roboflow(api_key=" ")
project = rf.workspace(" ").project(" ")
version = project.version(1)
dataset = version.download("yolov8")

#Load a Pretrained YOLOv8 Model
model = YOLO('yolov8n.pt')  # Start with the small model

#Train the Model
# The data argument now points to the YAML file inside your downloaded dataset
results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=50,  # Start with 50 epochs, you can increase this later
    imgsz=640
)


print("Training finished! Your best model is saved in the 'runs' folder.")
