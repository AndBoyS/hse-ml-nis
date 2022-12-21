FROM huggingface/transformers-pytorch-cpu:latest

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install gradio

# Make port 7860 available to the world outside this container
EXPOSE 7860

CMD ["python", "demo.py"]
