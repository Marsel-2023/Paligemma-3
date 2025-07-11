# Paligemma Image Recognition

## Описание

Веб-приложение на Flask для распознавания изображений с помощью модели Hugging Face (Gradio API). Позволяет загружать изображения, задавать текстовый промпт и получать результат анализа. Также на главной странице есть кнопка для запуска всех автотестов — результат тестов отображается прямо на странице.

---

## Возможности

- Загрузка изображения (JPG, JPEG, PNG, GIF) и текстового промпта через веб-интерфейс
- Получение результата анализа изображения от модели Hugging Face
- Кнопка **"Запустить тесты"** на главной странице:
  - Запускает все автотесты проекта
  - Результат (успешно/неуспешно, подробный вывод) отображается на странице

---

## Быстрый старт

1. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Запустите приложение:**
   ```bash
   python main.py
   ```
   После запуска откройте браузер и перейдите по адресу: [http://localhost:5000](http://localhost:5000)

3. **Использование:**
   - Загрузите изображение и введите промпт, нажмите "Распознать изображение" — результат появится ниже.
   - Для проверки корректности работы кода нажмите кнопку **"Запустить тесты"** — результат тестов появится на странице.

---

## Тесты

- Все тесты реализованы в файле `main.py` (используется pytest и unittest.mock).
- Кнопка "Запустить тесты" на главной странице запускает их автоматически и показывает результат.
- Для ручного запуска тестов из консоли:
  ```bash
  python main.py test
  ```

---

## Переменные окружения

- Для работы с Hugging Face API требуется токен. По умолчанию он захардкожен в коде, но рекомендуется использовать переменную окружения `HUGGING_FACE_TOKEN` (см. `.env` файл).

---
