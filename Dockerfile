FROM python:3.8-slim-buster

WORKDIR /fastapi-docker

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir -p /usr/local/nltk_data
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')"] 
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader omw-1.4
ADD ./ /fastapi-docker

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]