import os
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# Load YOLOv8 model (use custom if available)
model = YOLO("yolov8n.pt")  # Or path to your custom-trained model

# Input image folder (Food-101, for example)
input_dir = Path("/content/drive/MyDrive/Python Course_shared/computer Vision/food-101_stretched/images")  # Update this path
output_img_dir = Path("/content/drive/MyDrive/Python Course_shared/computer Vision/food-101_annotated/output")
output_csv_path = Path("/content/drive/MyDrive/Python Course_shared/computer Vision/output/food-101_annotated/annotations.csv")

# Create output directory
# output_img_dir.mkdir(parents=True, exist_ok=True)

# Prepare CSV file
csv_header = ['filename','predicted_class', 'x_min', 'y_min', 'x_max', 'y_max', ]
csv_rows = []
# Only include known "food" class names from COCO
FOOD_CLASSES = { "hot dog","pizza", "donut", "cake"
}
# Process images
for category in os.listdir(input_dir):
    class_dir = input_dir / category
    for img_file in os.listdir(class_dir):
        img_path = str(class_dir / img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Inference
        results = model(img)[0]

        # Draw and collect boxes
        # for box in results.boxes:
        #     x1, y1, x2, y2 = map(int, box.xyxy[0])
        #     conf = float(box.conf[0])
        #     class_id = int(box.cls[0])
        #     label = model.names[class_id]
            
        #     if label not in FOOD_CLASSES:
        #         continue

        #     # Draw bounding box
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     # Save to CSV
        #     csv_rows.append([img_file, x1, y1, x2, y2, conf, category])


        # Save annotated image
        out_path = output_img_dir/{category}/f"{img_file}"
        cv2.imwrite(str(out_path), img)

# Write CSV
output_csv_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

print(f"✅ Annotated images saved to: {output_img_dir}")
print(f"✅ CSV annotations saved to: {output_csv_path}")