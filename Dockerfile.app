# Dockerfile.app
FROM python:3.11-slim
WORKDIR /app
COPY requirements.app.txt ./
COPY pyproject.toml ./
COPY trainer ./trainer
RUN pip install --no-cache-dir -r requirements.app.txt \
	&& pip install --no-cache-dir .
COPY app ./app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
