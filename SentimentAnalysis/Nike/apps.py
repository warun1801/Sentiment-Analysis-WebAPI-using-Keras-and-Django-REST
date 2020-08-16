from django.apps import AppConfig
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from nltk import word_tokenize
from keras.preprocessing import sequence
import os


class NikeConfig(AppConfig):
    name = 'Nike'
    predictor = keras.models.load_model(
        os.getcwd()+"\\Nike\\sentiment_analysis.h5")