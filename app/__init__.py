from flask import Flask
from model import Model
from app.config import Config

app = Flask(__name__)
app.config.from_object(Config)

model = Model('models/LogisticRegressionCV.pickle')
model.load()

from app import routes