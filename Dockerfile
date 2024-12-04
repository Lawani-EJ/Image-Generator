FROM python:3.10
# This will specify the base image to use for the docker image
WORKDIR /app
# This sets the working directory for subsequent commands in the DockerFile
COPY requirements.txt requirements.txt
# This copies the files from the local file system to the DockerImage 
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "demo.py"]
# This specifies the command to run when the Docker container starts 