
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import numpy as np
import json

class Rapidapi():

    load_dotenv()

    def __init__(self,
            word_search='Bolsonaro',
            page_number='1',
            page_size='50',
            auto_correct='true',
            from_published_date='2019-01-01',
            to_published_date='2021-07-30'):

        """
            data_path: path of dsv file
            csv_name: nameof csv file
        """

        self.url = os.getenv('RAPIDAPI_URL')
        self.word_search = word_search
        self.page_number = str(page_number)
        self.page_size = str(page_size)
        self.auto_correct = auto_correct
        self.from_published_date = from_published_date
        self.to_published_date = to_published_date

        self.querystring = {"q":self.word_search,"pageNumber":self.page_number,"pageSize":self.page_size,"autoCorrect":self.auto_correct,"fromPublishedDate": self.from_published_date,"toPublishedDate": self.to_published_date}

        self.headers = {
            'x-rapidapi-key': os.getenv('X_RAPIDAPI_KEY'),
            'x-rapidapi-host': os.getenv('X_RAPIDAPI_HOST')
            }

    def request_api(self):
        self.response = requests.request("GET", self.url, headers=self.headers, params=self.querystring)


    def create_dataset(self):

        response_json = self.response.json()['value']

        self.df = pd.DataFrame(response_json)
        date_time = pd.to_datetime(self.df.datePublished)
        self.df['year'] = date_time.dt.year
        self.df['month'] = date_time.dt.month
        self.df = self.df[['id', 'title', 'body', 'year', 'month']].reset_index()
        self.df.columns = ['Unnamed: 0',  'id', 'title', 'content', 'year', 'month']



