import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset Iris
data = datasets.load_iris()
X = data.data
y = data.target

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Buat dan latih model Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=data.target_names)

# GUI Tkinter
root = tk.Tk()
root.title("Prediksi Jenis Bunga Iris")
root.geometry("420x550")
root.configure(bg="#f4f4f4")

style = ttk.Style()
style.configure("TButton", font=("Arial", 10, "bold"), padding=5)
style.configure("TLabel", font=("Arial", 10))

frame = ttk.Frame(root, padding=20)
frame.pack(pady=10)

# Input Fields
ttkk = ttk.Label(frame, text="Masukkan Data Bunga Iris", font=("Arial", 12, "bold"))
ttkk.grid(column=0, row=0, columnspan=2, pady=10)

entries = {}
fields = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
for i, field in enumerate(fields):
    ttk.Label(frame, text=field + ":").grid(column=0, row=i + 1, sticky="w")
    entry = ttk.Entry(frame, width=15)
    entry.grid(column=1, row=i + 1)
    entries[field] = entry

# Fungsi prediksi
def predict_flower():
    try:
        input_data = np.array([[float(entries[f].get()) for f in fields]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        result_label.config(text=f"Hasil Prediksi: {data.target_names[prediction[0]]}", foreground="blue")
    except ValueError:
        messagebox.showerror("Input Error", "Masukkan angka yang valid!")

# Fungsi menampilkan evaluasi model
def show_evaluation():
    eval_text = f"Akurasi: {accuracy * 100:.2f}%\n\nConfusion Matrix:\n{conf_matrix}\n\nLaporan Klasifikasi:\n{class_report}"
    messagebox.showinfo("Evaluasi Model", eval_text)

# Fungsi menampilkan Confusion Matrix
def show_confusion_matrix():
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

# Buttons
ttkk = ttk.Button(frame, text="Prediksi", command=predict_flower, style="TButton")
ttkk.grid(column=0, row=6, columnspan=2, pady=10)

result_label = ttk.Label(frame, text="", font=("Arial", 11, "bold"))
result_label.grid(column=0, row=7, columnspan=2, pady=5)

eval_button = ttk.Button(frame, text="Evaluasi Model", command=show_evaluation, style="TButton")
eval_button.grid(column=0, row=8, columnspan=2, pady=5)

conf_matrix_button = ttk.Button(frame, text="Confusion Matrix", command=show_confusion_matrix, style="TButton")
conf_matrix_button.grid(column=0, row=9, columnspan=2, pady=5)

root.mainloop()