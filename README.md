# 👕 Deep Learning Image Classification Web App

A web application for image classification using Convolutional Neural Networks (CNN) and transfer learning with VGG16.

The application :
- Processes user-uploaded images and adapts them to model-specific input requirements
- Performs real-time inference using CNN and transfer learning (VGG16)
- Outputs predicted class along with full probability distribution
- Displays model perfomance metrics (training/validation accuracy and loss)

Built with TensorFlow and Streamlit.

---

## 🚀 Features

- ✅ Custom CNN model trained on Fashion MNIST
- ✅ Transfer Learning with VGG16 (ImageNet weights)
- ✅ Fine-tuning of upper convolutional layers
- ✅ Interactive web interface (Streamlit)
- ✅ Visualization of training metrics (Accuracy & Loss)
- ✅ Probability distribution for all classes
- ✅ Support for two model selection modes (CNN / VGG16)

## 🧠 Models

### 1️⃣ Custom CNN
- Trained on Fashion MNIST (28x28 grayscale)
- Convolution + Pooling layers
- Fully connected classifier

### 2️⃣ VGG16 Transfer Learning
- Pretrained on ImageNet
- Input resized to 80x80
- 3-channel conversion
- Fine-tuning of block5 layers
- GlobalAveragePooling + Dense classifier

---

## 📊 Dataset
Fashion MNIST  
10 classes:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## 🖥️ Demo Interface

The web app allows users to:

1. Select model (CNN or VGG16)
2. Upload an image (PNG/JPG)
3. View predicted class
4. See probability distribution
5. Inspect training Accuracy & Loss curves

---

## 🛠️ Tech Stack

- Python 3.13
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit
- Pillow

---

## ⚙️ Installation

Clone repository:
1. Make sure you have Python 3.13 installed
2. Clone the repository:
```bash
git clone https://github.com/your-username/project-name.git
```
3. Navigate to the project folder:
``` bash
cd project-name
```
4. (Optional) Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
5. Install dependencies:
```bash
pip install -r requirements.txt
```
## 🏃 Run Apllication
To run Image Classification Web Application :
```bash
streamlit run app.py
```