import string
from datetime import date

import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
from dotenv import load_dotenv
import os
import numpy as np
import emoji

class DataProcessor(object):

    load_dotenv()

    def __init__(self, csv_path='../raw_data/', csv_name='articles1'):
        """
            data_path: path of dsv file
            csv_name: nameof csv file
        """
        self.csv_path = csv_path
        self.csv_name = csv_name

    def load_sql(self, table, start_date='2000-01-01', end_date='2030-01-01'):

        """
            load data from sql database
        """
        query = {'table': table, 'start_date':start_date, 'end_date':end_date}
        response = requests.get(os.getenv('SQL_URL'), params=query)

        if (response.status_code == 200):
            df = pd.read_json(response.json())
            return df_.copy()

        print("Result not found!")
        return None

    def save_sql(self, table, data):

        """
            save data in sql database
        """
        query = {'table': table, 'data':data}
        response = requests.post(os.getenv('SQL_URL'), data=query)

        if (response.status_code == 200):
            return response.json()

        print("Result not found!")
        return None

    def load_bucket(self, filename, extension='csv', nrows=None):

        """
            load data from bucket
        """

        params = {'filename': filename, 'extension': extension, 'nrows': nrows}
        response = requests.get(os.getenv('GETSTORAGE_URL'), params=params)
        if (response.status_code == 200):
            df = pd.read_json(response.json())
            return df_.copy()

        print("Result not found!")

        return None

    def save_bucket(self, filename, data, extension='csv'):

        """
            save data on bucket
        """

        query = {'filename': filename, 'extension': extension, 'data':data}

        response = requests.get(os.getenv('POSTSTORAGE_URL'), params=query)

        if (response.status_code == 200):
            return response.json()

        return {'error': "Result not found!"}

    def del_bucket(self, filename):

        """
            delete data on bucket
        """

        response = requests.delete(f"{os.getenv('POSTSTORAGE_URL')}/{filename}")

        if (response.status_code == 200):
            return response.json()

        return {'error': "Result not found!"}
        
    def load_dataset(self):
        """
            load dataset
        """
        
        df_ = pd.read_csv(self.csv_path + self.csv_name + '.csv', sep=',')
        #df_['date'] = pd.to_datetime(df_['date'])
        return df_.copy()

    def load_npy(self):
        """
            load dataset
        """
        df = np.load(f"{self.csv_path}{self.csv_name}.npy")
        return df.copy()
        
    def load_dataset_processed(self):
        """
            load the dataset processed
        """
        df_ = pd.read_csv(self.csv_path + self.csv_name + '_preproced.csv', sep=',')
        #df_['date'] = pd.to_datetime(df_['date'])
        return df_.copy()

    def __remove_punctuation(self, text):
        """
            remove punctuation from text and lower case it
        """
        text = str(text)

        punctuations = string.punctuation
        punctuations += '“'
        punctuations += '’'
        punctuations += '”'
        punctuations += '’'
        punctuations += ' — '
        punctuations += 'â€œ'
        punctuations += 'â€¦'
        punctuations += 'â€'
        punctuations += '€™'
        punctuations += '€'
        punctuations += '™'
        punctuations += '¦'
        punctuations += 'œ'
        punctuations += 'Â'
        punctuations += 'Ã'
        punctuations += '— '
        punctuations += '¶'
        punctuations += '§'
        punctuations += '£'
        punctuations += '©'
        punctuations += 'ª'
        punctuations += '³'

        text = emoji.get_emoji_regexp().sub(u'', text)

        for punctuation in punctuations:
            text = text.replace(punctuation, ' ') 
            #text = text.replace('donald', 'trump')
            #text = text.replace('clinton', 'hillary')
        return text.lower() # lower case

    def __remove_numbers(self, text):
        """
            remove number from text
        """
        text = str(text)

        words_only = ''.join([i for i in text if not i.isdigit()])
        return words_only

    def __remove_stopwords(self, text):
        """
            remove stop words from text
        """
        text = str(text)

        stop_words = stopwords.words('english')
        #stop_words += stopwords.words('portuguese')
        stop_words.append('mr')
        stop_words = set(stop_words)

        tokenized = word_tokenize(text)
        without_stopwords = [word for word in tokenized if not word in stop_words]
        return without_stopwords

    def __lemmatize(self, text):
        """
            lemmatize text
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in text]
        lemmatized_string = " ".join(lemmatized)
        return lemmatized_string

    def __customsentiment(self, text, custom_values):

        for key, value in custom_values.items():
            new_text = []
            for t in text.split(" "):
                t = value if t==key else t
                new_text.append(t)

        return " ".join(new_text)

    def process_data(self):
        """
            process the data
        """
        df_ = self.load_dataset()
        #df_.drop(columns='url', inplace=True)
        #df_['author'] = df_['author'].fillna('no_author')
        df_['title'] = df_['title'].fillna('no_title')
        df_['title_preproced'] = df_['title'].copy()

        #date_today = date.today()
        #df_['date'] = df_['date'].fillna(date_today)

        #df_['date'] = pd.to_datetime(df_['date'])
        #df_['year'] = df_['date'].dt.year
        #df_['month'] = df_['date'].dt.month
        
        df_ = df_.reset_index(drop=True)
        
        df_['title_preproced'] = df_['title_preproced'].apply(self.__remove_punctuation)
        df_['content'] = df_['content'].apply(self.__remove_punctuation)

        df_['title_preproced'] = df_['title_preproced'].apply(self.__remove_numbers)
        df_['content'] = df_['content'].apply(self.__remove_numbers)

        df_['title_preproced'] = df_['title_preproced'].apply(self.__remove_stopwords)
        df_['content'] = df_['content'].apply(self.__remove_stopwords)

        df_['title_preproced'] = df_['title_preproced'].apply(self.__lemmatize)
        df_['content'] = df_['content'].apply(self.__lemmatize)
        
        df_.to_csv(self.csv_path + self.csv_name + '_preproced.csv', sep=',', index=False)

if __name__ == '__main__':
    dp = DataProcessor(csv_path='../raw_data/', csv_name='articles1')
    df = dp.load_dataset_processed()
