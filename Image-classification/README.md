# 🖼️ Image Classification Project

## 📌 Overview
This project implements an image classification system using a Convolutional Neural Network (CNN) trained on the CIFAR dataset.  
It includes:
- A trained model (`model_cifar.h5`)
- A Python GUI (`gui.py`) for easy interaction

## 🚀 Features
- Classifies images into CIFAR categories
- Simple GUI for uploading and testing images
- Pre-trained model for quick predictions

## 🛠️ Installation
Clone the repository and set up the environment:

```bash
git clone https://github.com/anushsubba101/ml-learning-journey.git
cd ml-learning-journey/Image-classification
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt

Run the GUI:
'python gui.py'

📂 Project Structure
Image-classification/
│── gui.py              # GUI application
│── model_cifar.h5      # Trained CNN model
│── venv/               # Virtual environment (ignored in repo)
│── requirements.txt    # Dependencies

📸 Demo
![alt text](Image-classification/image.png)
![alt text](Image-classification/img2.png)
