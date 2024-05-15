FROM python:3.11-buster

ENV PYTHONUNBUFFERED=1

# create code directory
RUN mkdir /code
WORKDIR /code

# copy the requirement file and install packages
COPY ./requirements.txt /code/

# upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /code/

EXPOSE 8501 6006
STOPSIGNAL SIGINT

