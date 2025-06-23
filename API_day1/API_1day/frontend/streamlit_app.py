import streamlit as st
import requests
from PIL import Image
import io

st.title("🚀 Классификатор изображений и текста")
st.write("Это приложение использует ResNet18 для классификации изображений и DistilBERT для классификации текста")

tab1, tab2 = st.tabs(["Классификация изображений", "Классификация текста"])

with tab1:
    st.header("Классификация изображений (ResNet18)")
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        if st.button("Классифицировать изображение"):
            # Отправка изображения на сервер
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://localhost:8000/classify_image/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Результат классификации: {result['class']} (ID: {result['class_id']})")
            else:
                st.error("Ошибка при обработке изображения")

with tab2:
    st.header("Классификация текста (DistilBERT)")
    text_input = st.text_area("Введите текст для анализа настроения", "I love this movie!")
    
    if st.button("Анализировать текст"):
        # Отправка текста на сервер
        response = requests.post(
            "http://localhost:8000/classify_text/",
            json={"text": text_input}
        )
        
        if response.status_code == 200:
            result = response.json()
            sentiment = "😊 Положительный" if result['label'] == "POSITIVE" else "😞 Отрицательный"
            st.success(f"Настроение: {sentiment} (точность: {result['score']:.2f})")
        else:
            st.error("Ошибка при анализе текста")

st.sidebar.markdown("### Информация")
st.sidebar.info("""
Это приложение использует:
- FastAPI как backend
- Streamlit как frontend
- ResNet18 для классификации изображений
- DistilBERT для анализа настроения текста
""")