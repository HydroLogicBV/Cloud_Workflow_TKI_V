FROM python:3.9-slim-bullseye

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    && pip install contextily --no-cache 
# anders doet contextily het niet (met wat voor reden weet ik niet)

COPY ./app /code/app

CMD ["python", "app/main.py"]