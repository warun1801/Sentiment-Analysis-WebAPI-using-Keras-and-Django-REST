from django.shortcuts import render
from .apps import NikeConfig

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from nltk import word_tokenize
from keras.preprocessing import sequence


class call_model(APIView):

    def get(self, request):
        if request.method == 'GET':

            # sentence is the query we want to get the prediction for
            params = request.GET.get('sentence')
            text = str(params)
            # print(sentence)

            max_len = 200
            word2index = imdb.get_word_index()
            test = []
            for word in word_tokenize(text):
                if word2index[word] <= 25000:
                    test.append(word2index[word])

            test = sequence.pad_sequences([test], maxlen=max_len)
            # predict method used to get the prediction
            response = NikeConfig.predictor.predict(test)
            if response > 0.4:
                response = {"status": "Negative", "val": float(response)}
            else:
                response = {"status": "Positive", "val": float(response)}
            # returning JSON response
            return JsonResponse(response, safe=False)
