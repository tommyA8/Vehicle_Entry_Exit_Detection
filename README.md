# Data Collection and Annotation
- Using [Roboflow](https://roboflow.com/) for annotating images and splitting the data into Training, Validation, and Testing sets.
here is my custom dataset https://app.roboflow.com/tommya8-yyxnu/cars-detection-lq124/4
- 4 Classes: `['bike', 'bus', 'car', 'truck']`
#
# Fine-tuning Yolov8n trained on COCO
Using `200 Epochs` with `AdamW`

<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/yolov8n_custom_model/runs_datasetv4/detect/train2/confusion_matrix_normalized.png?raw=true" width="800" height="600"/>
<p align="center">Confusion Matrix Normalized

# Vehicle Entry and Exit Detection
-  Resize to 640x640 for fitting a model <p align="left">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/yolov8n_custom_model/runs_datasetv4/detect/train2/confusion_matrix_normalized.png?raw=true" width="400" height="300"/></p><p align="left">Confusion Matrix Normalized</p>