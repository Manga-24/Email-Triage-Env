FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py tasks.py env.py server.py openenv.yaml ./

EXPOSE 7860

CMD ["python", "server.py"]
