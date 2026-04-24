# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем наш код Flask-приложения
COPY app.py .

# Открываем порт 5000
EXPOSE 5000

# Команда для запуска нашего API
CMD ["python", "app.py"]