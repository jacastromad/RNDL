FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update -y
RUN apt-get -y install python3-tk
RUN pip install -r requirements.txt

