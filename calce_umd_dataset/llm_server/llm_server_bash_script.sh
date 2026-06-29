#!/bin/bash

# ==============================================================================
# Скрипт развертывания локального LLM-координатора (Qwen2.5-3B)
# ==============================================================================

echo "=== 1. Инициализация виртуального окружения ==="
if [ ! -d "llm_venv" ]; then
    python3 -m venv llm_venv
    echo "Виртуальное окружение 'llm_venv' создано."
else
    echo "Виртуальное окружение 'llm_venv' уже существует."
fi

source llm_venv/bin/activate

echo "=== 2. Установка зависимостей ==="
pip install --upgrade pip
# Устанавливаем huggingface-cli для быстрой загрузки моделей
pip install huggingface_hub
# Устанавливаем llama.cpp сервер (OpenAI API совместимый)
# Если у вас есть видеокарта NVIDIA, раскомментируйте строку ниже перед запуском скрипта для GPU-акселерации:
# CMAKE_ARGS="-DGGML_CUDA=on" pip install 'llama-cpp-python[server]' --upgrade --force-reinstall --no-cache-dir
pip install 'llama-cpp-python[server]'

echo "=== 3. Загрузка весов модели Qwen2.5-3B-Instruct (GGUF, Q5_K_M) ==="
MODEL_DIR="./models"
mkdir -p $MODEL_DIR
MODEL_FILE="qwen2.5-3b-instruct-q5_k_m.gguf"
REPO_ID="Qwen/Qwen2.5-3B-Instruct-GGUF"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Скачивание модели из HuggingFace (около 2.4 ГБ)..."
    huggingface-cli download $REPO_ID $MODEL_FILE --local-dir $MODEL_DIR --local-dir-use-symlinks False
else
    echo "Модель $MODEL_FILE уже найдена в директории $MODEL_DIR. Скачивание пропущено."
fi

echo "=== 4. Запуск локального API-сервера ==="
echo "Сервер будет доступен по адресу: http://localhost:8001"
echo "API эндпоинт: http://localhost:8000/v1/chat/completions"
echo "Для остановки сервера нажмите Ctrl+C"
echo "--------------------------------------------------------"

# Запускаем сервер с контекстом 4096 токенов
python3 -m llama_cpp.server \
    --model "$MODEL_DIR/$MODEL_FILE" \
    --host 0.0.0.0 \
    --port 8001 \
    --n_ctx 4096
