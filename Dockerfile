FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and the serialized model
# Notice: In a real CI/CD pipeline, the model is downloaded from MLflow
# before building the image, and placed into the 'model' directory.
COPY src/ /app/src/
COPY model/ /app/model/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model

EXPOSE 5000

# Use gunicorn for production serving
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]
