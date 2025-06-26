import streamlit as st
from gradio_client import Client, file
import warnings
import os

# Настройки предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)


@st.cache_resource
def get_client():
    # Загружаем токен из переменной окружения
    haggi_token = os.getenv("HUGGING_FACE_TOKEN")

    if not haggi_token:
        raise ValueError("Токен не найден. Убедитесь, что переменная окружения HUGGING_FACE_TOKEN установлена.")

    # Инициализация клиента
    return Client("big-vision/paligemma")


# Используем кэшированный клиент
client = get_client()


def analyze_image(image_path, prompt):
    try:
        result = client.predict(
            file(image_path),
            prompt,
            "paligemma-3b-mix-448",  # Модель
            "greedy",  # Алгоритм декодирования
            api_name="/compute"
        )
        token_value = result[0]['value'][0]['token']
        return token_value
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return None


def main():
    st.title("Распознавание изображении - Paligemma от Google")

    st.write("Загрузите изображение и введите текстовый промпт для анализа.")

    # Форма загрузки файла
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Введите текстовый промпт")

    if st.button("Распознать"):
        if uploaded_file and prompt:
            # Сохранение загруженного файла
            temp_file_path = f"uploaded_image.jpg"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Анализ изображения
            st.write("Обрабатываем изображение...")
            token_value = analyze_image(temp_file_path, prompt)

            if token_value:
                st.success(f"На фотографии: {token_value}")
        else:
            st.error("Пожалуйста, загрузите изображение и введите промпт.")


if __name__ == "__main__":
    main()
