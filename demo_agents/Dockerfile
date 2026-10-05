FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --uid 10001 --create-home appuser
USER appuser
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "main.py"]
