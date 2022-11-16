FROM python:3.8

RUN apt-get update
RUN apt-get install libgl1 -y

WORKDIR /app

ENV FLASK_APP=app.py

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "/app" ]