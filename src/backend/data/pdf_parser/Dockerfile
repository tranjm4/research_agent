FROM python:3.11

WORKDIR /app

# Import dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./pdf_parser/parser.py .
COPY ./message_queue/message_queue.py ./message_queue/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "parser.py"]