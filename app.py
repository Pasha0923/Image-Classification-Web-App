import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
st.title("Fashion MNIST Classification App")

# Назви класів
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Вибір моделі
model_option = st.selectbox("Вибрати модель:", ("CNN", "VGG16"))

# Завантаження моделі CNN і history
if model_option == "CNN":
    model = load_model("model_cnn.keras")
    history = np.load("history_cnn.npy", allow_pickle=True).item()
else:
# Завантаження моделі VGG16 і history
    model = load_model("VGG16_for_app.keras", compile=False)
    history = np.load("history_vgg_for_app.npy", allow_pickle=True).item()

# Вивід графіків
if st.checkbox("Показати графіки функції втрат і точності"):
    st.subheader("Графіки навчання моделі")
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    # Accuracy
    ax[0].plot(history['accuracy'], label='Train Accuracy')
    ax[0].plot(history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title("Accuracy")
    ax[0].legend()
    # Loss
    ax[1].plot(history['loss'], label='Train Loss')
    ax[1].plot(history['val_loss'], label='Val Loss')
    ax[1].set_title("Loss")
    ax[1].legend()
    st.pyplot(fig)


# Функції передобробки зображення
def preprocess_cnn(image):
    image = image.convert("L") 
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (28,28,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,28,28,1)
    return img_array

def preprocess_vgg(image):
    image = image.convert("L")
    image = image.resize((80, 80))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.stack([img_array]*3, axis=-1)  # (80,80,3)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array


# Завантаження зображення
uploaded_file = st.file_uploader("Завантажте зображення", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", width="stretch")

    # Передобробка
    if model_option == "CNN":
        img = preprocess_cnn(image)
    else:
        img = preprocess_vgg(image)

    # Передбачення
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

   
    # Результати класифікації
    st.subheader("Результати класифікації")
    st.success(f"Передбачений клас: {class_names[predicted_class]}")
    st.write("Ймовірності по класам:")
    prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)
