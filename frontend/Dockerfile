FROM ubuntu:20.04
ADD . /app
WORKDIR /app
RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN python3.9 -m pip install --upgrade setuptools
RUN pip3 install django gunicorn
RUN python3.9 -m pip install -r requirements.txt
RUN python3.9 -m pip install psycopg2-binary

 
EXPOSE 8000     