FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
# models/ is created at runtime; no copy needed
ENV ACCURACY_THRESHOLD=0.90
CMD ["python", "src/train_and_evaluate.py"]
