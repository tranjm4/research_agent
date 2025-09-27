FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./harvester/harvester.py .
COPY ./message_queue/message_queue.py ./message_queue/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "harvester.py"]