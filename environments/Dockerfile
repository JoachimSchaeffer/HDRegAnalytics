# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get -y install texlive-latex-extra texlive-fonts-recommended dvipng \
&& apt-get -y install cm-super \
&& apt-get -y install ghostscript \
&& apt-get clean 

RUN python -m pip install -r requirements.txt

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" HDfeatlin-user && chown -R HDfeatlin-user /home
RUN useradd -ms /bin/bash HDfeatlin-user
USER HDfeatlin-user

WORKDIR /home/HDfeatlin-user
COPY . /home/HDfeatlin-user