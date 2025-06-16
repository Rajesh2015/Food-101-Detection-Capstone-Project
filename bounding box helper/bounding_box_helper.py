import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def draw_yolo_bboxes_from_csv(data_dir, image_file, annotations_df):
    images_dir = os.path.join(data_dir, 'images')
    image_path = os.path.join(images_dir, image_file)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Image not found: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Filter annotations for this image
    boxes = annotations_df[annotations_df['image'] == image_file]

    if boxes.empty:
        print(f"⚠️ No annotations found for: {image_file}")
        return

    # Draw each bounding box
    for _, row in boxes.iterrows():
        class_id = row['class_name']
        x_center = float(row['x_center']) * width
        y_center = float(row['y_center']) * height
        w = float(row['width']) * width
        h = float(row['height']) * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Draw rectangle and class_id
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_rgb, str(class_id), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"{image_file}")
    plt.show()