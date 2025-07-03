from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash
from gradio_client import Client, file, handle_file
import warnings
import os
import sys
import pytest
import tempfile
from unittest.mock import Mock, patch, mock_open
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Настройки предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)

# Создаем Flask приложение
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def get_client():
    # Загружаем токен из переменной окружения
    haggi_token = os.environ.get("TOKEN_HUGGI")
    if not haggi_token:
        raise ValueError("Токен не найден. Убедитесь, что переменная окружения TOKEN_HUGGI установлена.")

    # Инициализация клиента
    return Client("amd/llama4-maverick-17b-128e-mi-amd")


# Глобальный клиент (будет инициализирован при первом использовании)
client = None


def get_or_create_client():
    global client
    if client is None:
        client = get_client()
    return client


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(image_path, prompt):
    try:
        client = get_or_create_client()
        result = client.predict(
            message={"text": prompt, "files": [handle_file(image_path)]},
            param_2="",
            param_3=2048,
            param_4=0.3,
            param_5=0,
            param_6=0,
            api_name="/chat"
        )

        token_value = result
        return {"success": True, "result": token_value}
    except Exception as e:
        return {"success": False, "error": str(e)}


# HTML шаблон для главной страницы
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Распознавание изображений - Paligemma</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="file"], input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        input[type="text"] { margin-bottom: 10px; }
        button { background-color: #4CAF50; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .loading { background-color: #cce5ff; border: 1px solid #99ccff; color: #004085; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 5px; }
        .upload-area:hover { border-color: #999; }
        .test-btn { background-color: #007bff; margin-top: 10px; }
        .test-btn:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Распознавание изображений - Paligemma</h1>
        
        <form method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image">Загрузите изображение:</label>
                <div class="upload-area">
                    <input type="file" id="image" name="image" accept=".jpg,.jpeg,.png,.gif" required>
                    <p>Поддерживаемые форматы: JPG, JPEG, PNG, GIF</p>
                </div>
            </div>
            
            <div class="form-group">
                <label for="prompt">Текстовый промпт:</label>
                <input type="text" id="prompt" name="prompt" placeholder="Например: 'Что изображено на картинке?' или 'Describe this image'" required>
            </div>
            
            <button type="submit">🔍 Распознать изображение</button>
        </form>
        <button class="test-btn" id="run-tests-btn">🧪 Запустить тесты</button>
        <div id="test-result"></div>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="result {{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% if result %}
            <div class="result success">
                <strong>Результат анализа:</strong><br>
                {{ result }}
            </div>
        {% endif %}
    </div>
    
    <script>
        // Показать имя файла после выбора
        document.getElementById('image').addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                const uploadArea = document.querySelector('.upload-area p');
                uploadArea.textContent = `Выбран файл: ${fileName}`;
            }
        });
        // Кнопка запуска тестов
        document.getElementById('run-tests-btn').addEventListener('click', function() {
            const btn = this;
            btn.disabled = true;
            btn.textContent = '⏳ Тесты выполняются...';
            document.getElementById('test-result').innerHTML = '';
            fetch('/run_tests', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('test-result').innerHTML = `<div class='result success'>✅ Все тесты пройдены успешно!</div>`;
                    } else {
                        document.getElementById('test-result').innerHTML = `<div class='result error'>❌ Некоторые тесты не прошли.<br>\n${data.output || ''}</div>`;
                    }
                })
                .catch(e => {
                    document.getElementById('test-result').innerHTML = `<div class='result error'>Ошибка запуска тестов</div>`;
                })
                .finally(() => {
                    btn.disabled = false;
                    btn.textContent = '🧪 Запустить тесты';
                });
        });
    </script>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем, что файл загружен
        if 'image' not in request.files:
            flash('Файл не был загружен', 'error')
            return redirect(request.url)

        file = request.files['image']
        prompt = request.form.get('prompt', '').strip()

        # Проверяем, что файл выбран
        if file.filename == '':
            flash('Файл не был выбран', 'error')
            return redirect(request.url)

        # Проверяем, что промпт введен
        if not prompt:
            flash('Введите текстовый промпт', 'error')
            return redirect(request.url)

        # Проверяем расширение файла
        if file and allowed_file(file.filename):
            try:
                # Сохраняем файл
                filename = secure_filename(file.filename)
                temp_file_path = f"temp_{filename}"
                file.save(temp_file_path)

                # Анализируем изображение
                result = analyze_image(temp_file_path, prompt)

                # Удаляем временный файл
                try:
                    os.remove(temp_file_path)
                except:
                    pass

                if result["success"]:
                    return render_template_string(HTML_TEMPLATE, result=result["result"])
                else:
                    flash(f'Ошибка анализа: {result["error"]}', 'error')

            except Exception as e:
                flash(f'Ошибка обработки файла: {str(e)}', 'error')
        else:
            flash('Неподдерживаемый формат файла. Используйте JPG, JPEG, PNG или GIF', 'error')

    return render_template_string(HTML_TEMPLATE)


@app.route('/run_tests', methods=['POST'])
def run_tests_api():
    """Endpoint для запуска тестов через фронт"""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = run_tests()
        output = buf.getvalue()
        return jsonify({
            "success": result == 0,
            "output": output
        })
    except Exception as e:
        return jsonify({"success": False, "output": str(e)})


# =================== ТЕСТЫ ===================

class TestGetClient:
    """Тесты для функции get_client"""

    @patch.dict(os.environ, {'TOKEN_HUGGI': 'test_token'})
    @patch('main.Client')
    def test_get_client_with_token_success(self, mock_client_class):
        """Тест успешного создания клиента с токеном"""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        result = get_client()

        mock_client_class.assert_called_once_with("amd/llama4-maverick-17b-128e-mi-amd")
        assert result == mock_client_instance

    @patch.dict(os.environ, {'TOKEN_HUGGI': 'test_token'})
    @patch('main.Client')
    def test_get_or_create_client_caching(self, mock_client_class):
        """Тест кэширования клиента"""
        global client
        client = None  # Сбрасываем глобальный клиент

        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Первый вызов
        result1 = get_or_create_client()
        # Второй вызов
        result2 = get_or_create_client()

        mock_client_class.assert_called_once_with("amd/llama4-maverick-17b-128e-mi-amd")
        assert result1 == result2 == mock_client_instance


class TestAnalyzeImage:
    """Тесты для функции analyze_image"""

    @patch('main.get_or_create_client')
    @patch('main.handle_file')
    def test_analyze_image_success(self, mock_handle_file, mock_get_client):
        """Тест успешного анализа изображения"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_handle_file.return_value = "mocked_file"
        mock_client.predict.return_value = "test_result"

        result = analyze_image("test_image.jpg", "test prompt")

        mock_handle_file.assert_called_once_with("test_image.jpg")
        mock_client.predict.assert_called_once_with(
            message={"text": "test prompt", "files": ["mocked_file"]},
            param_2="",
            param_3=2048,
            param_4=0.3,
            param_5=0,
            param_6=0,
            api_name="/chat"
        )
        assert result == {"success": True, "result": "test_result"}

    @patch('main.get_or_create_client')
    @patch('main.handle_file')
    def test_analyze_image_client_error(self, mock_handle_file, mock_get_client):
        """Тест обработки ошибки клиента"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_handle_file.return_value = "mocked_file"
        mock_client.predict.side_effect = Exception("API Error")

        result = analyze_image("test_image.jpg", "test prompt")

        assert result == {"success": False, "error": "API Error"}

    @patch('main.get_or_create_client')
    @patch('main.handle_file')
    def test_analyze_image_invalid_response_format(self, mock_handle_file, mock_get_client):
        """Тест обработки неверного формата ответа (неактуально, т.к. теперь возвращается строка)"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_handle_file.return_value = "mocked_file"
        mock_client.predict.return_value = None

        result = analyze_image("test_image.jpg", "test prompt")

        assert result["success"] is True
        assert result["result"] is None

    @patch('main.get_or_create_client')
    @patch('main.handle_file')
    def test_analyze_image_empty_prompt(self, mock_handle_file, mock_get_client):
        """Тест с пустым промптом"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_handle_file.return_value = "mocked_file"
        mock_client.predict.return_value = "empty_result"

        result = analyze_image("test_image.jpg", "")

        mock_client.predict.assert_called_once_with(
            message={"text": "", "files": ["mocked_file"]},
            param_2="",
            param_3=2048,
            param_4=0.3,
            param_5=0,
            param_6=0,
            api_name="/chat"
        )
        assert result == {"success": True, "result": "empty_result"}


class TestFlaskApp:
    """Тесты для Flask приложения"""

    @pytest.fixture
    def test_client(self):
        """Создает тестовый клиент Flask"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_get_homepage(self, test_client):
        """Тест загрузки главной страницы"""
        response = test_client.get('/')
        assert response.status_code == 200
        assert "Распознавание изображений - Paligemma" in response.data.decode()

    def test_post_without_file(self, test_client):
        """Тест POST запроса без файла"""
        response = test_client.post('/', data={'prompt': 'test prompt'}, follow_redirects=True)
        assert "Файл не был загружен" in response.data.decode()

    def test_post_without_prompt(self, test_client):
        """Тест POST запроса без промпта"""
        # Создаем временный файл изображения
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake_image_data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                data = {
                    'image': (f, 'test.jpg'),
                    'prompt': ''
                }
                response = test_client.post('/', data=data, follow_redirects=True)
                assert "Введите текстовый промпт" in response.data.decode()
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    def test_post_invalid_file_type(self, test_client):
        """Тест POST запроса с неподдерживаемым типом файла"""
        # Создаем временный файл с неподдерживаемым расширением
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"fake_text_data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                data = {
                    'image': (f, 'test.txt'),
                    'prompt': 'test prompt'
                }
                response = test_client.post('/', data=data, follow_redirects=True)
                # Проверяем, что показывается сообщение об ошибке
                assert "Неподдерживаемый формат файла" in response.data.decode()
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    @patch('main.analyze_image')
    @patch('os.remove')
    def test_post_successful_analysis(self, mock_remove, mock_analyze, test_client):
        """Тест успешного анализа изображения"""
        mock_analyze.return_value = {"success": True, "result": "test_result"}

        # Создаем временный файл изображения
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake_image_data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                data = {
                    'image': (f, 'test.jpg'),
                    'prompt': 'test prompt'
                }
                response = test_client.post('/', data=data)

                assert response.status_code == 200
                assert "test_result" in response.data.decode()
                mock_analyze.assert_called_once()
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass

    @patch('main.analyze_image')
    @patch('os.remove')
    def test_post_failed_analysis(self, mock_remove, mock_analyze, test_client):
        """Тест неудачного анализа изображения"""
        mock_analyze.return_value = {"success": False, "error": "API Error"}

        # Создаем временный файл изображения
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake_image_data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                data = {
                    'image': (f, 'test.jpg'),
                    'prompt': 'test prompt'
                }
                response = test_client.post('/', data=data, follow_redirects=True)

                # Проверяем, что показывается сообщение об ошибке
                assert "Ошибка анализа: API Error" in response.data.decode()
                mock_analyze.assert_called_once()
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass


class TestUtilityFunctions:
    """Тесты для вспомогательных функций"""

    def test_allowed_file_valid_extensions(self):
        """Тест проверки разрешенных расширений файлов"""
        assert allowed_file('test.jpg') == True
        assert allowed_file('test.jpeg') == True
        assert allowed_file('test.png') == True
        assert allowed_file('test.gif') == True
        assert allowed_file('TEST.JPG') == True  # case insensitive

    def test_allowed_file_invalid_extensions(self):
        """Тест проверки неразрешенных расширений файлов"""
        assert allowed_file('test.txt') == False
        assert allowed_file('test.pdf') == False
        assert allowed_file('test.exe') == False
        assert allowed_file('test') == False  # no extension
        assert allowed_file('') == False  # empty filename


class TestIntegration:
    """Интеграционные тесты"""

    @patch.dict(os.environ, {'TOKEN_HUGGI': 'test_token'})
    @patch('main.Client')
    def test_full_workflow_mock(self, mock_client_class):
        """Интеграционный тест полного workflow с моками"""
        global client
        client = None  # Сбрасываем глобальный клиент

        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.predict.return_value = "mocked_result"

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(b"fake_image_data")
            temp_file_path = temp_file.name

        try:
            # Тестируем полный workflow
            client_instance = get_or_create_client()
            result = analyze_image(temp_file_path, "test prompt")

            assert result == {"success": True, "result": "mocked_result"}
            mock_client.predict.assert_called_once()
        finally:
            # Очистка
            os.unlink(temp_file_path)

    @pytest.fixture
    def test_client(self):
        """Создает тестовый клиент Flask для интеграционных тестов"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    @patch('main.analyze_image')
    def test_full_flask_workflow(self, mock_analyze, test_client):
        """Интеграционный тест полного Flask workflow"""
        mock_analyze.return_value = {"success": True, "result": "integration_test_result"}

        # Создаем временный файл изображения
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake_image_data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                data = {
                    'image': (f, 'integration_test.jpg'),
                    'prompt': 'integration test prompt'
                }
                response = test_client.post('/', data=data)

                assert response.status_code == 200
                assert "integration_test_result" in response.data.decode()
                mock_analyze.assert_called_once()
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass


def run_flask_app():
    """Запуск Flask приложения"""
    print("🌐 Запуск Flask приложения...")
    print("Откройте браузер и перейдите по адресу: http://localhost:5000")
    print("Для остановки приложения нажмите Ctrl+C")

    # Проверяем наличие токена
    try:
        get_client()
        print("✅ Токен Hugging Face найден")
    except ValueError as e:
        print(f"❌ {e}")
        print("Установите переменную окружения TOKEN_HUGGI")
        return

    # Запускаем Flask приложение
    app.run(host='0.0.0.0', port=5000, debug=False)


def run_tests():
    """Запуск всех тестов с подробным выводом"""
    print("🧪 Запуск тестов...")
    print("=" * 50)

    # Настройка pytest
    pytest_args = [
        __file__,  # Текущий файл
        "-v",  # Подробный вывод
        "-s",  # Показать print
        "--tb=short",  # Короткий traceback
        "--no-header",  # Без заголовка
        "-q",  # Тихий режим для менее загроможденного вывода
    ]

    # Запуск тестов
    result = pytest.main(pytest_args)

    if result == 0:
        print("\n✅ Все тесты пройдены успешно!")
    else:
        print(f"\n❌ Некоторые тесты не прошли (код выхода: {result})")

    return result


if __name__ == "__main__":
    # Запуск Flask приложения сразу, без меню
    run_flask_app()
