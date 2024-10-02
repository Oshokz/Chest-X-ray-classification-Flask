# Use an official Python runtime as the base image
FROM python:3.9.19

# This will bring your Flask application files, requirements.txt, and other project files into the container
COPY . /app

# Set the working directory in the container to /app
# This is where the application files will be stored in the container
WORKDIR /app

# Install the dependencies specified in requirements.txt
# This installs all the necessary Python packages for your Flask app to run
RUN pip install -r requirements.txt

# Set the command to run your Flask application
CMD python app.py