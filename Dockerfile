FROM python:3.9 AS torch

# Install dependencies
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install gradio transformers

FROM torch

ENV GRADIO_SERVER_NAME=0.0.0.0

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Make port 7860 available to the world outside this container
EXPOSE 7860

CMD ["python", "demo.py"]
