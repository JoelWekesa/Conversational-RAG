FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt ./README.md ./

COPY ./package[s] ./packages

COPY ./app ./app

COPY ./docs ./docs

RUN pip install -r requirements.txt

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8000
