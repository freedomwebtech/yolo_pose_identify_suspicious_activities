import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math

# Initialize the YOLO model and video capture
model = YOLO('yolov8s-pose.pt')
cap = cv2.VideoCapture('vid5.mp4')

count = 0

labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Define the area for drawing polylines
area1 = [(712, 198), (556,391), (752, 464), (849, 214)]
angle = []
sperson = {}
suspecious={}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 2 != 0:
        continue
    
    # Make predictions
    result = model.track(frame)
    
    if result[0].boxes is not None and result[0].boxes.id is not None:
        boxes = result[0].boxes.xyxy.int().cpu().tolist()
        keypoints = result[0].keypoints.xy.cpu().numpy()
        track_ids = result[0].boxes.id.int().cpu().tolist()
        conf = result[0].boxes.conf.cpu().tolist()
        
        if len(keypoints) > 0:
            for box, t_id, keypoint, confidence in zip(boxes, track_ids, keypoints, conf):
                # Only process persons with valid track IDs
                if t_id is None:
                    continue
                
                conf_pe = int(confidence * 100)
                if conf_pe > 50:   
                    cx1 = cy1 = cx2 = cy2 = cx3 = cy3 = None
                    for j, point in enumerate(keypoint):
                        label = labels[j]
                        x1, y1, x2, y2 = box
                        
                        if j == 6:  # Left shoulder
                            cx, cy = int(point[0]), int(point[1])
                            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                        if j == 8:  # Left hip
                            cx1, cy1 = int(point[0]), int(point[1])
                            cv2.circle(frame, (cx1, cy1), 4, (255, 0, 0), -1)
                            cv2.line(frame, (cx, cy), (cx1, cy1), (255, 0, 255), 2)

                        if j == 10:  # Left elbow
                            cx2, cy2 = int(point[0]), int(point[1])
                            cv2.circle(frame, (cx2, cy2), 4, (255, 0, 0), -1)
                            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)

                            v1 = np.array([cx, cy]) - np.array([cx1, cy1])
                            v2 = np.array([cx2, cy2]) - np.array([cx1, cy1])
                            dot_product = np.dot(v1, v2)
                            norm_v1 = np.linalg.norm(v1)
                            norm_v2 = np.linalg.norm(v2)

                            # Prevent division by zero and clip value to avoid invalid arccos input
                            if norm_v1 != 0 and norm_v2 != 0:
                                cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
                                angle_rad = np.arccos(cos_angle)
                                angle_degree = np.degrees(angle_rad)
                                angle.append(angle_degree)
                                angle_text = f'Angle: {angle_degree:.2f}'
                                cvzone.putTextRect(frame, f'{angle_text}', (50, 60), 1, 1)
                                sperson[t_id] = (cx2, cy2)
                                result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx2, cy2), False)

                                if result >= 0:
                                   if 100 <= angle_degree <= 113:
                                       suspecious[t_id]=(cx2,cy2)   
                                       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                       cvzone.putTextRect(frame, f'ID: {t_id}', (x1, y1), 1, 1)

                                if t_id in sperson and t_id in suspecious:  # Only continue if t_id is in sperson
                                   if 90 <= angle_degree <= 144:
                                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                      cvzone.putTextRect(frame, f'ID: {t_id}', (x1, y1), 1, 1)
                                else:
                                      cvzone.putTextRect(frame, f'ID: {t_id}', (x1, y1), 1, 1)

                                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)

                            
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    # Display the frame
    cv2.imshow("RGB", frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
