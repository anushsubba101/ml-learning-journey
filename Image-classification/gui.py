# =========================
# CIFAR Image Classifier GUI
# =========================

# Install if needed:
# pip install tensorflow pillow

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------
# Load Trained Model
# -------------------------
# Use either .keras or .h5
model = load_model("model_cifar.h5")

# -------------------------
# CIFAR-10 Class Names
# -------------------------
classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# -------------------------
# Predict Function
# -------------------------
def predict_image(filepath):

    # Open image
    img = Image.open(filepath).convert("RGB")

    # Resize to CIFAR input size
    img = img.resize((32, 32))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
    )

# -------------------------
# Upload Image Function
# -------------------------
def upload_image():

    filepath = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if filepath:

        # Display image
        img = Image.open(filepath)
        img = img.resize((250, 250))

        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predict
        predict_image(filepath)

# -------------------------
# GUI Window
# -------------------------
root = tk.Tk()
root.title("CIFAR Image Classifier")
root.geometry("500x500")
root.configure(bg="white")

# Title
title = tk.Label(
    root,
    text="Image Classifier",
    font=("Arial", 20, "bold"),
    bg="white"
)
title.pack(pady=10)

# Image display
image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

# Upload button
upload_btn = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14),
    bg="blue",
    fg="white",
    padx=10,
    pady=5
)
upload_btn.pack(pady=10)

# Result label
result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 14),
    bg="white"
)
result_label.pack(pady=20)

# Run GUI
root.mainloop()