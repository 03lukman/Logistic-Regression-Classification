# Iris Flower Classification with Logistic Regression

## 📌 About the Project
This project is a **Supervised Learning** implementation for **Multiclass Classification** using **Logistic Regression**. The goal is to classify Iris flower species (**Setosa, Versicolor, and Virginica**) based on four key features:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

The model is trained on the **Iris dataset** from Scikit-learn, which consists of **150 samples** with a balanced distribution among the three classes.

To enhance user interaction, the project includes a **Tkinter-based GUI** that allows users to input flower features and get real-time predictions. Additionally, it provides model evaluation through **accuracy, confusion matrix, and classification report**, along with data visualization using **Seaborn and Matplotlib**.

---

## 🚀 Features
✅ **Train and test a Logistic Regression model** on the Iris dataset  
✅ **Interactive GUI** using Tkinter for real-time predictions  
✅ **Model evaluation metrics**: Accuracy, Confusion Matrix, Classification Report  
✅ **Data visualization** with Seaborn and Matplotlib  
✅ **User-friendly input form** for feature entry  

---

## 🛠 Tech Stack
- **Python**
- **Scikit-learn** (Machine Learning model)
- **Tkinter** (GUI)
- **Matplotlib & Seaborn** (Data visualization)
- **Pandas & NumPy** (Data handling)

---

## 📂 Project Structure
```
📁 iris-flower-classification
│── 📄 main.py            # Main script containing GUI and model
│── 📄 README.md          # Project documentation
│── 📄 requirements.txt   # Dependencies list
```

---

## 🔧 Installation & Usage
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
python main.py
```

---

## 📊 Model Evaluation
**Accuracy Score:** `{:.2f}%`

**Confusion Matrix:**
```
[[TN  FP]
 [FN  TP]]
```

**Classification Report:**
```
Precision    Recall    F1-score    Support
------------------------------------------------
Setosa         1.00        1.00        1.00        50
Versicolor     0.96        0.96        0.96        50
Virginica      0.98        0.98        0.98        50
------------------------------------------------
Overall Accuracy: {:.2f}%
```

---

## 📌 Screenshots
### 🌟 GUI Interface
![GUI Screenshot](screenshot.png)

### 📊 Confusion Matrix Visualization
![Confusion Matrix](confusion_matrix.png)

---

## 👨‍💻 Author
- **[Your Name]**  
- [LinkedIn Profile](https://www.linkedin.com/in/yourprofile)  
- [GitHub Profile](https://github.com/your-username)

---

## ⭐ Acknowledgments
This project was created as part of **Digital Skill Fair 36.0 - Data Science** by **dibimbing.id**.

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify it!

