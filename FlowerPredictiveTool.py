import tkinter as tk
from tkinter import filedialog
from tkinter import *
from keras.models import load_model
from PIL import Image, ImageOps
from PIL import Image, ImageTk
import numpy as np

def flower_Class():
    np.set_printoptions(suppress=True)
    model = load_model("Flowers\keras_model.h5", compile=False)
    class_names = open("Flowers\labels.txt", "r").readlines()
    image_path = filedialog.askopenfilename()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    img = Image.open(image_path)
    img = img.resize((250,250),Image.LANCZOS)
    image = ImageTk.PhotoImage(img)
    imageLabel.configure(image = image)
    imageLabel.image = image

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    info.insert(tk.END, f"Class: {class_name}\n")
    info.insert(tk.END, f"Confidence score: {confidence_score}")

root = tk.Tk()
root.title("Flower Image Classifier")
frame = tk.Frame(root)
frame.pack(side=TOP,padx=15, pady=15)
lbl=Label(root)
lbl.pack()
imageLabel = tk.Label(frame)

title = tk.Label(frame, text = "Flower Image Classifier", bg = 'light blue', font= 14)
title.pack(side=tk.TOP)

image1 = tk.Button(frame, text="Select Image", command=flower_Class)
image1.pack(side=tk.LEFT)

button = tk.Button(frame, text="QUIT", fg="red", command=quit)
button.pack(side=tk.RIGHT)

info = tk.Text(frame, height= 3, width= 40)
info.pack(side=tk.BOTTOM)

imageLabel.pack(side = tk.TOP)
root.geometry("500x500")
root.mainloop()




