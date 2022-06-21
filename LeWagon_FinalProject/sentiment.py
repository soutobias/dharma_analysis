from LeWagon_FinalProject.data import *
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import InputExample, InputFeatures
from scipy.special import softmax
import numpy as np
import pandas as pd

from transformers import pipeline

class SentimentLarge():
    def __init__(self, X):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.model = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")


    def predict(self, *args):
        self.pred = []
        if args:
            text_size = int(args[0])
        else:
            text_size = -1
        for idx, row in self.X.items():
            print(idx)
            self.pred.append(self.model(row[0:text_size])[0])

        self.pred = pd.DataFrame(self.pred)

class Sentiment():
    def __init__(self, X):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.labels = ['negative', 'neutral', 'positive']
        self.processor = DataProcessor()
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.x = 0

    def sentiment_analisys(self, *args, **kwargs):
        if kwargs:
            self.preprocess()
        if args:
            text_size = args[0]
            self.encoder(text_size)
        else:
            self.encoder()

        pred = []
        for index, value in self.X.items():
            output = self.model(**self.encoded_input[index])
            scores = output[0][0].detach().numpy()
            scores = (softmax(scores))
            if pred == []:
                pred = scores
            else:
                pred = np.vstack((pred, scores))

        self.pred = pd.DataFrame(pred, columns=self.labels)

    def encoder(self, *args):
        if args:
            text_size = int(args[0])
        else:
            text_size = 500
        print(text_size)
        self.encoded_input = self.X.apply(lambda text: self.tokenizer(text[0:text_size], return_tensors='pt'))

    def preprocess(self, custom_values):
        self.X = self.X.apply(lambda x: self.processor.__customsentiment(custom_values))


if __name__ == "__main__":
    X = pd.Series(['Good',
                  'One of the worst movies of all time'])

    sentiment = Sentiment(X)

    sentiment.sentiment_analisys()
    print(sentiment.pred)
