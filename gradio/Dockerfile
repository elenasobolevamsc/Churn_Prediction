FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY frontend.py .

EXPOSE 7880

# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7880"

# Run the Gradio app
CMD ["python", "frontend.py"]