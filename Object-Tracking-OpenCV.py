# Tracking => compare the coordinates of objects in their current frame and previous frame => if the distance between coordinates is less than 20 => assign a specific track_id to the object

# libraries -----------------------------------------------------
import cv2
from ultralytics import YOLO
import math
# libraries -----------------------------------------------------


# Detector models -----------------------------------------------------
Vehicle_detector = YOLO('yolov8n.pt')  
# Detector models -----------------------------------------------------


cap = cv2.VideoCapture("./data/video_test.mp4")
vehicles = [2, 3, 5, 7] # only detect cars, motorbikes, buses, trucks,


count = 0
tracking_objects = {}
track_id = 0
center_points_previous =[]


while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    center_points_current = []


# detect vehicles-----------------------------------------------
    detections_v = Vehicle_detector(frame)[0]
    detectioned_v= []

    for detection in detections_v.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detectioned_v.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)   # darw a box around detected objects

# detect vehicles-----------------------------------------------


# Track vehicles-----------------------------------------------
        # the center of each object
            cx = int((x1 + x2)/ 2)
            cy = int((y1 + y2)/ 2)

        # the center of each object in each frame
            center_points_current.append((cx, cy))
    
    # compare distances between coordinates in current & previous frames + assign a track Id to each object 
    if count <= 2:
        for pt in center_points_current:
            for pt2 in center_points_previous:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) # claculate the distance between coordinates of objects in current & previous frames
                if distance < 20: # if the distance is less that 20, then they are the same objects => so assign a track id to the objects
                    tracking_objects[track_id] = pt
                    track_id += 1

    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_current_copy = center_points_current.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_current_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # update ids position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_current:
                        center_points_current.remove(pt)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id) # if the object is nor exist anymore remove it from tracking_objects

        for pt in center_points_current:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 0.5, (0, 0, 255), 1)
# Track vehicles-----------------------------------------------



    print("Tracking objects--------------------------------------")
    print(tracking_objects)

    print("Current points--------------------------------------")
    print(center_points_current)

    print("Previous points--------------------------------------")
    print(center_points_previous)

    # fill the "center_points_previous" with the pevious "center_points_current" in each frame
    center_points_previous = center_points_current.copy()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()





