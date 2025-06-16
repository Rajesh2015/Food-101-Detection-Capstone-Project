import os
import base64
import requests
import csv
import time

# === CONFIGURATION ===
API_KEY = 'key'
IMAGE_DIR = 'images/'  # directory with images
OUTPUT_CSV = 'yolo_annotations.csv'
API_URL = 'https://api.openai.com/v1/chat/completions'
MODEL = 'gpt-4-vision-preview'

# === HEADERS ===
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

# === PREPARE CSV ===
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'class_name', 'x_center', 'y_center', 'width', 'height'])

    # === LOOP THROUGH IMAGES ===
    for image_file in os.listdir(IMAGE_DIR):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(IMAGE_DIR, image_file)
        print(f'Processing {image_file}...')

        # === READ AND ENCODE IMAGE ===
        with open(image_path, 'rb') as img_f:
            image_bytes = img_f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # === PROMPT ===
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Please analyze this image and return object annotations in YOLO format. "
                            f"Use the format: image_name class_name x_center y_center width height (normalized between 0 and 1). "
                            f"Return only plain text. The image name is {image_file}."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # === SEND REQUEST ===
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1000,
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']

            # === PARSE LINES ===
            for line in content.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 6:
                    writer.writerow(parts)
                else:
                    print(f"⚠️ Skipping line: {line}")
            
            time.sleep(2)  # respectful delay

        except Exception as e:
            print(f"❌ Failed for {image_file}: {e}")
