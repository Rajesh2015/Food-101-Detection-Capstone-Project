# 🥘 Capstone Project - Food Identification using Computer Vision

**📚 Provided by:** Great Learning
**⚠️ Notice:** Proprietary content. All Rights Reserved. Unauthorized use or distribution is prohibited.

---

## 🧠 Problem Statement

### 📌 Domain: Food Industry

Computer vision can be used to automate supervision and trigger appropriate actions when events are predicted from images. For instance, cameras can identify food items by analyzing:

* Type of food
* Color
* Ingredients
  This has wide applications in food tech, quality control, and smart kitchens.

---

## 📊 Data Description

* Dataset: **[Food101](https://www.kaggle.com/datasets/dansbecker/food-101)**
* Contains: **16,256 images** of **17 food classes**
* Structure: Each class can be roughly split into **70% training** and **30% testing**
* Example class: `apple_pie`, `chicken_curry`, etc.
* 💾 The dataset is **already provided** with this project. Use only the provided version.

📖 **Reference Paper**:
*Food-101 – Mining Discriminative Components with Random Forests*
by Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool

---

## 🎯 Project Objective

Design a **Deep Learning-based food identification system** using classification and object detection models.

---

## ✅ Project Tasks

### 🧩 Milestone 1 

**Input**: Raw dataset and context.
**Steps**:

1. **Import the dataset** 
2. **Map training/testing images to food classes** 
3. **Create annotations** for 10 food classes

   * Pick **10 classes**, choose **50 images** from each
   * Manually annotate images using any image annotation tool 
4. **Display images with bounding boxes** created in step 3 
5. **Build, train, and test a basic CNN** for food classification 
6. **Submit interim report** 

**📦 Submission**:

* Interim Report
* Jupyter Notebook with all Milestone 1 steps

---

### 🧩 Milestone 2 

**Input**: Preprocessed data from Milestone 1
**Steps**:

1. **Fine-tune basic CNN models** 
2. **Build & test RCNN and hybrid object detection models**

   * Use bounding boxes or masks for identifying regions of interest 
3. **Pickle the trained model for future predictions** 
4. **Submit final report** 

**📦 Submission**:

* Final Report
* Jupyter Notebook with Milestone 1 + Milestone 2

---

### 🧪 Milestone 3 – \[Optional Bonus]

Design a **clickable UI interface**:

* Allow users to browse & input images
* Output predicted **class** and **bounding box/mask**

**📦 Submission**:

* Final Report
* Jupyter Notebook with UI interface

---

## 🔗 References & Learning Resources

* [Object Detection Using TensorFlow (Great Learning)](https://www.mygreatlearning.com/blog/object-detection-using-tensorflow/)
* [YOLO Object Detection with OpenCV](https://www.mygreatlearning.com/blog/yolo-object-detection-using-opencv/?highlight=detection)
* [Face Detection Techniques](https://www.mygreatlearning.com/blog/face-recognition/?highlight=detection)
* For GUI (desktop): `Tkinter`
* For GUI (web): `Flask` or `Django`

---
Stream_It Deployted Model
https://food-101-capstone-cv5.streamlit.app/
