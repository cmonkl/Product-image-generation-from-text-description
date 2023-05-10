FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /var/www

WORKDIR /var/www
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /var/www/requirements.txt
