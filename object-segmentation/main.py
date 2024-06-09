import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('marathon_race.mp4')
model = YOLO('yolov8s-seg.pt')

# Getting the video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Saving the segmented video
out = cv2.VideoWriter('marathon_segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform object segmentation
        results = model(frame)
        annotated_frame = results[0].plot()

        # Write the annotated frame to the video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow('Object segmentation', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
