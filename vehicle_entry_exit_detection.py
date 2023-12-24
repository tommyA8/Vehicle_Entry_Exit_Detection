from ultralytics import YOLO
import cv2
import datetime
import numpy as np
import pandas as pd
from tracker import Tracker

# Load a model
check_point = 'train2'
model_path = f'./yolov8n_custom_model/runs_datasetv4/detect/{check_point}/weights/best.pt'
model = YOLO(model_path)  # load a custom model

# Load a video
video_name = 'IMG_4599'
video_path = f'./videos/{video_name}.MOV'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# export a video
output_path = f'./outputs/OUTPUT_{video_name}.mp4'
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (490, 300)) #

def plot_boxes():

    if class_id == 0: color = (0, 100, 255) # orange
    elif class_id == 1: color = (255, 255, 0) # Cyan
    elif class_id == 2: color = (0, 255, 0) # green
    else: color = (255, 0, 255) # Magenta

    cv2.rectangle(frame, (x, y), (w, h), color, 2)
    cv2.circle(frame,(cx,cy),4,(color),-1)

    pred_tracking = f"id:{id} {results.names[int(class_id)]}{'%.2f' % score}"
    text_size, _ = cv2.getTextSize(pred_tracking, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x, y-10), (x+text_w, y-10+text_h+5), color, -1)
    cv2.putText(frame, pred_tracking, (x, y-10+text_h), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)

def crop_frame(frame):
    roi_x, roi_y, roi_width, roi_height = 50, 270, resize_w-100, resize_h-70
    #cv2.rectangle(frame, (roi_x, roi_y), (roi_width, roi_height), (0,0,0), 2)
    roi = frame[roi_y:roi_height, roi_x:roi_width]
    h, w, _ = roi.shape
    return roi, h, w 

count_in = 0
count_out = 0
already_in_IDs = set()
already_out_IDs = set()
data = {
    "Date-time": [],
    "Class_in": [],
    "Class_out": [],
}

# resize 
resize_w, resize_h = 640, 640

# classifies confidence
threshold = 0.4

tracker = Tracker()

while True:
    
    ret, frame = cap.read()

    if not ret:
        break

    # resize image for fitting model
    frame = cv2.resize(frame, (640, 640))
    height, width, _ = frame.shape

    # crop
    frame, height, width = crop_frame(frame)

    verticle_line = width//2
    horizontal_line = height//2

    """
    vehicle detecting
    """
    # precdict (object detection work here)
    results = model(frame, verbose=False)[0]

    # date-time when vehicle was detected
    date_time = str(datetime.datetime.now())

    detectionsByFrame = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if score > threshold:
            detectionsByFrame.append([x1, y1, x2, y2, score, class_id])

    """
    tracking
    """
    tracking_results = tracker.update(detectionsByFrame)

    for track_result in tracking_results:
        x, y, w, h, score, class_id, id = track_result
        x, y, w, h = int(x), int(y), int(w), int(h)
        cx, cy = (x + w)//2, (y + h)//2
        plot_boxes()
        """
        Entering and leaving
        """
        offset = 15
        # Entering
        if cx > verticle_line and cy > horizontal_line and cy < horizontal_line + offset:
            # check if vehicle already in
            if id not in already_in_IDs:
                count_in += 1
                data['Date-time'].append(date_time)
                data['Class_in'].append(results.names[int(class_id)])
                data['Class_out'].append(None)
            # if not, add id into 'already_in_IDs' set
            already_in_IDs.add(id)

        # Leaving
        if cx < verticle_line and cy > horizontal_line and cy < horizontal_line + offset:
            # check if vehicle already out
            if id not in already_out_IDs:
                count_out += 1
                data['Date-time'].append(date_time)
                data['Class_in'].append(None)
                data['Class_out'].append(results.names[int(class_id)])
            # if not, add id into 'already_in_IDs' set
            already_out_IDs.add(id)

    """
    show info.
    """
    info_color = (255, 255, 255)
    info_thickness = 2
    # Number of vehicle IN and OUT
    cv2.putText(frame, f"IN:{count_in}", (verticle_line+5, horizontal_line-5),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, info_color, info_thickness)
    cv2.putText(frame, f"OUT:{count_out}", (0, horizontal_line-5),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, info_color, info_thickness)
    # counting region
    overlay = frame.copy()
    cv2.rectangle(frame, (0, horizontal_line), (width, horizontal_line + offset), info_color, -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0)
    cv2.line(frame, (verticle_line, 0), (verticle_line, height), info_color, info_thickness) # vertical line
    # show date time
    cv2.putText(frame, date_time, (15, 20), 
                 cv2.FONT_HERSHEY_DUPLEX, 0.6, info_color, info_thickness)
    # show frame
    cv2.imshow('vehicle_entry_exit_detection', frame)

    cap_out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cap_out.release()
cv2.destroyAllWindows()

# show DataFrame (Output)
df = pd.DataFrame(data)
# print(df.info)
df.to_csv(f'./outputs/vehicle_entry_exit_detection_{video_name}.csv', index=False)

