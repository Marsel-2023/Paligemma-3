import sys
import os
import streamlit as st
from unittest.mock import patch, MagicMock

# Проверка наличия pytest перед импортом
try:
    import pytest
except ImportError:
    print("Ошибка: Модуль pytest не установлен.")
    print("Установите его с помощью команды: pip install pytest pytest-mock")
    sys.exit(1)

# Решение проблемы импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Импорт после настройки пути
try:
    from main import analyze_image, get_client, main
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print(f"sys.path: {sys.path}")
    raise

# Тесты для get_client
def test_get_client(monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "test_token")
    client = get_client()
    assert client is not None

def test_get_client_no_token(monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    with pytest.raises(ValueError):
        get_client()

# Тесты для analyze_image
@patch('main.Client')
def test_analyze_image_success(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.predict.return_value = [{'value': [{'token': 'cat'}]}]
    
    result = analyze_image("test.jpg", "What is this?")
    assert result == "cat"

@patch('main.Client')
def test_analyze_image_failure(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.predict.side_effect = Exception("API error")
    
    result = analyze_image("test.jpg", "What is this?")
    assert result is None

# Тесты для основного интерфейса
@patch('main.analyze_image')
def test_main_success(mock_analyze, monkeypatch):
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(st, 'text_input', lambda *args, **kwargs: "test")
    monkeypatch.setattr(st, 'button', lambda *args, **kwargs: True)
    monkeypatch.setattr(st, 'success', MagicMock())
    
    mock_analyze.return_value = "result"
    main()
    
    mock_analyze.assert_called_once()
    st.success.assert_called_with("На фотографии: result")

@patch('main.analyze_image')
def test_main_missing_inputs(mock_analyze, monkeypatch):
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: None)
    monkeypatch.setattr(st, 'text_input', lambda *args, **kwargs: "")
    monkeypatch.setattr(st, 'button', lambda *args, **kwargs: True)
    monkeypatch.setattr(st, 'error', MagicMock())
    
    main()
    
    mock_analyze.assert_not_called()
    st.error.assert_called_with("Пожалуйста, загрузите изображение и введите промпт.")

@patch('main.analyze_image')
def test_main_analyze_error(mock_analyze, monkeypatch):
    """Тест обработки ошибки анализа изображения"""
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(st, 'text_input', lambda *args, **kwargs: "test prompt")
    monkeypatch.setattr(st, 'button', lambda *args, **kwargs: True)
    monkeypatch.setattr(st, 'error', MagicMock())
    
    mock_analyze.return_value = None
    main()
    
    mock_analyze.assert_called_once()
    st.error.assert_called()

# Запуск тестов при прямом выполнении файла
if __name__ == "__main__":
    # Проверка наличия pytest
    try:
        import pytest
    except ImportError:
        print("Ошибка: Модуль pytest не установлен.")
        print("Установите его с помощью команды: pip install pytest pytest-mock")
        sys.exit(1)
    
    # Запуск тестов
    result = pytest.main(["-v", __file__])
    sys.exit(result)