FROM python:3-onbuild

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /var/app
COPY ./ .
COPY requirements.txt .

RUN python -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet")'

EXPOSE 5000

CMD ["gunicorn", "web_classifier:app", "--bind", "0.0.0.0:5000"]