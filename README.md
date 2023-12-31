# Data collection and Annotation
- Using [Roboflow](https://roboflow.com/) for annotating images and splitting the data into Training, Validation, and Testing sets.
here is my custom dataset https://universe.roboflow.com/tommya8-yyxnu/cars-detection-lq124
- 4 Classes: `['bike', 'bus', 'car', 'truck']`
#
# Fine-tuning Yolov8n trained on COCO
- Using `200 Epochs` with `AdamW` optimizer

<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/yolov8n_custom_model/runs_datasetv4/detect/train2/confusion_matrix_normalized.png?raw=true" width="700" height="500"/>
<p align="center">Confusion Matrix Normalized

# Vehicle Entry and Exit Detection
- Resize to 640x640 for fitting a model 
<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/images/resize.jpeg?raw=true" width="350" height="350"/>

- Crop the image to enhance detection accuracy
<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/images/crop.jpeg?raw=true" width="490" height="300"/>

- Vehicles are detected and kept tracked across frames
<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/images/detected_tracked.jpeg?raw=true" width="490" height="300"/>

- Count the number of vehicles that enter/exit the ROI
<p align="center">
<img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/images/counting.jpeg?raw=true" width="490" height="300"/>

- Create a `DataFrame` for storing entering and exiting vehicles by class and recording time.
    ```python
    data = {
        "Date-time": [],
        "Class_in": [],
        "Class_out": [],
    }
    df = pd.DataFrame(data)
    df.to_csv(f'./outputs/vehicle_entry_exit_detection_{video_name}.csv', index=False)
    ```
    
# Final Result
OUTPUT_IMG_4498.GIF            |  OUTPUT_IMG_4499.GIF
:-------------------------:|:-------------------------:
![IMG_4498](https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/outputs/OUTPUT_IMG_4598_GIF.gif?raw=true)  |  ![IMG_4499](https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/outputs/OUTPUT_IMG_4599_GIF.gif?raw=true)


- Output table `OUTPUT_TABLE_IMG_4498.csv`

    <p align="center">
    <img src="https://github.com/tommyA8/Vehicle_Entry_Exit_Detection/blob/main/outputs/OUTPUT_TABLE_IMG_4498.jpg?raw=true"/>
