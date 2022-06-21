import pandas as pd
import numpy as np
import os

class PrepareGraphDataset(object):

    def __init__(self, file_path, file_path_title=''):
        ''' file_path: path GenerateGraphData saved datasets '''

        self.file_path = file_path
        self.file_path_title  = file_path_title

    def prepare_most_important_center(self, df_topic, keep_documents_no_topic = False):
        ''' Sort topics, most important topic in the center '''
        
        df_topic = df_topic.copy()

        # remove topic -1
        if df_topic['Topic'].iloc[0] == -1:
            df_no_topic = df_topic.head(1).copy()
            df_topic.drop(index=0, inplace=True)
            df_topic.reset_index(inplace=True, drop=True)
        else:
            keep_documents_no_topic = False

        # get most important topic according number of news
        df_first = df_topic.head(1).copy()

        # remove most important topic according number of news
        df_topic.drop(index=0, inplace=True)
        df_topic.reset_index(inplace=True, drop=True)

        # get even rows
        df_even = df_topic.iloc[::2].copy()  # even
        df_even.sort_values(by='Topic', ascending=False, inplace=True)

        # get odd rows
        df_odd = df_topic.iloc[1::2]  # odd

        # concatenate even rows with most important topic (higher number of news)
        df_even = pd.concat([df_even, df_first], ignore_index=True)

        # concatenate with odd rows
        df_even = pd.concat([df_even, df_odd], ignore_index=True)
        
        if keep_documents_no_topic:
            df_even = pd.concat([df_even, df_no_topic], ignore_index=True)

        return df_even.copy()

    def prepare_higher_probabilities_center(self, df_docs):
        ''' Sort documents, most likely documents in the center '''
        
        df_docs = df_docs.copy()
        if (df_docs[df_docs['probabilities'] == 1].shape[0] <= 0) or (df_docs[df_docs['probabilities'] == 1].shape[0] == df_docs.shape[0]):
            return df_docs.copy()

        # get most likely probabilities
        df_center = df_docs[df_docs['probabilities'] == 1].copy()
        df_center.sort_values(by='document', ascending=True, inplace=True)

        # remove most likely probabilities
        df_docs = df_docs[df_docs['probabilities'] != 1].copy()
        df_docs.reset_index(inplace=True, drop=True)

        # get even rows
        df_even = df_docs.iloc[::2].copy()  # even
        df_even.sort_values(by='probabilities', ascending=True, inplace=True)

        # get odd rows
        df_odd = df_docs.iloc[1::2]  # odd
        df_odd.sort_values(by='probabilities', ascending=False, inplace=True)

        # concatenate even rows with most important topic (higher number of news)
        df_even = pd.concat([df_even, df_center], ignore_index=True)

        # concatenate with odd rows
        df_even = pd.concat([df_even, df_odd], ignore_index=True)
        
        return df_even.copy()

    def generate_graph_dataset(self, keep_documents_no_topic=False, similarity_threshold=0.6):
        ''' Based on the datasets below, creates a new dataset to be input for graph (Hilbert curve)
            keep_documents_no_topic=False: documents of topic -1 will not be taken into consideration
            keep_documents_no_topic=True: documents of topic -1 will be taken into consideration
            similarity_threshold: similarity threshold to define document1 size
            list_of_years_months.csv: dataset with the prefix of below datasets
            BERTopic_Info.csv
            BERTopic_TopicSimilarity.npy
            BERTopic_DocumentsSimilarity.npy
            HDBSCAN_TopicDocuments.csv
            sentiment.csv
            dataset.csv
        '''

        file_name = self.file_path + 'list_of_years_months.csv'
        df_prefix = pd.read_csv(file_name)
        for _, row in df_prefix.iterrows():
            print(row["year_month"])
            prefix = row["year_month"]
            
            topic_info_exists = False
            try:
                #file_name = self.file_path + prefix + 'BERTopic_Info.csv'
                file_name = self.file_path + prefix + 'BERTopic_Info_reduction.csv'
                if not os.path.exists(file_name):
                    file_name = self.file_path + prefix + 'BERTopic_Info.csv'
                df_topic_info = pd.read_csv(file_name)
                if df_topic_info.shape[0] > 1:
                    topic_info_exists = True
                else:
                    print(f'There are no topics for {prefix}')
            except:
                print(f'There are no topics for {prefix}')
            
            if topic_info_exists:
                # Sort topics, most important one in the midle of Hilbert curve
                df_topic_info = self.prepare_most_important_center(df_topic_info, keep_documents_no_topic)
                df_topic_info['similarity_previous_topic'] = 0.0
                
                # Cosine similarity to previous topic
                file_name = self.file_path + prefix + 'BERTopic_TopicSimilarity.npy'
                matrix_topics_similarity = np.load(file_name)        
                for ind_row, topic_row in df_topic_info.iterrows():
                    if ind_row > 0:
                        df_topic_info['similarity_previous_topic'].iloc[ind_row] = matrix_topics_similarity[int(topic_row['Topic'])+1, int(previous_topic)+1]
                    previous_topic = topic_row['Topic']
                
                # Topic documents
                file_name = self.file_path + prefix + 'BERTopic_DocumentsSimilarity.npy'
                matrix_documents_similarity = np.load(file_name)
                
                #file_name = self.file_path + prefix + 'HDBSCAN_TopicDocuments.csv'
                #df_topic_documents_hdbscan = pd.read_csv(file_name)
                file_name = self.file_path + prefix + 'BERTopic_TopicDocuments_reduction.csv'
                if not os.path.exists(file_name):
                    file_name = self.file_path + prefix + 'BERTopic_TopicDocuments.csv'
                df_topic_documents = pd.read_csv(file_name)
                df_topic_documents.rename(columns={'document_id': 'document'}, inplace=True)
                
                file_name = self.file_path + prefix + 'sentiment.csv'
                df_semtiment = pd.read_csv(file_name)
                df_semtiment['temp'] = df_semtiment['neutral']+df_semtiment['positive']-df_semtiment['negative']
                temp_max = df_semtiment['temp'].max()
                temp_min = df_semtiment['temp'].min()
                df_semtiment['setiment_scale'] = (df_semtiment['temp'] - temp_min) / (temp_max - temp_min)
                df_semtiment.drop(['temp'], axis=1, inplace=True)

                file_name = self.file_path + prefix + 'dataset.csv'
                df_dataset = pd.read_csv(file_name)

                if self.file_path_title != '':
                    file_name = self.file_path_title + prefix + 'dataset.csv'
                    df_dataset_title = pd.read_csv(file_name)

                for ind_row, topic_row in df_topic_info.iterrows():
                    top = topic_row['Topic']
                    #df_temp = df_topic_documents_hdbscan[df_topic_documents_hdbscan['topic']==top].sort_values(by=['probabilities'], ascending=True).reset_index(drop=True)
                    df_temp = df_topic_documents[df_topic_documents['topic']==top].reset_index(drop=True)
                    #df_temp = self.prepare_higher_probabilities_center(df_temp)
                    df_temp['probabilities'] = 1
                    df_temp['document1_title'] = topic_row['Name']
                    df_temp['document1_size'] = 1
                    df_temp['document2'] = 0.0
                    df_temp['similarity_previous_document'] = 0.0
                    df_temp['similarity_previous_topic'] = 0.0
                    df_temp['topic_name'] = 'topic'
                    df_temp['sentimet_classification'] = 'neutral'
                    df_temp['sentimet_score'] = 0.5
                    df_temp['setiment_scale'] = 0.0
                    for ind_temp_row, temp_row in df_temp.iterrows():
                        doc_ind = int(temp_row['document'])
                        if ind_temp_row > 0:
                            df_temp['similarity_previous_document'].iloc[ind_temp_row] = matrix_documents_similarity[doc_ind, int(previous_document)]
                            df_temp['document2'].iloc[ind_temp_row] = previous_document
                        else:
                            df_temp['document2'].iloc[ind_temp_row] = temp_row['document']
                        df_temp['similarity_previous_topic'] = topic_row['similarity_previous_topic']
                        df_temp['topic_name'] = topic_row['Name']
                        previous_document = temp_row['document']
                        
                        # Setiment classification                        
                        negative = df_semtiment['negative'].iloc[doc_ind]
                        neutral = df_semtiment['neutral'].iloc[doc_ind]
                        positive = df_semtiment['positive'].iloc[doc_ind]
                        if (neutral >= negative) and (neutral >= positive):
                            sentimet_classification = 'neutral'
                            sentimet_score = neutral
                        elif (negative > neutral) and (negative > positive):
                            sentimet_classification = 'negative'
                            sentimet_score = negative
                        else:
                            sentimet_classification = 'positive'
                            sentimet_score = positive
                        df_temp['sentimet_classification'].iloc[ind_temp_row] = sentimet_classification
                        df_temp['sentimet_score'].iloc[ind_temp_row] = sentimet_score
                        df_temp['setiment_scale'].iloc[ind_temp_row] = df_semtiment['setiment_scale'].iloc[doc_ind]
                        
                        doc_size = len(matrix_documents_similarity[doc_ind][matrix_documents_similarity[doc_ind] >= similarity_threshold])
                        df_temp['document1_size'].iloc[ind_temp_row] = doc_size
                        if self.file_path_title != '':
                            df_temp['document1_title'].iloc[ind_temp_row] = df_dataset_title['title'].iloc[doc_ind]
                        else:
                            df_temp['document1_title'].iloc[ind_temp_row] = df_dataset['title'].iloc[doc_ind]

                    df_temp.rename(columns={'document': 'document1'}, inplace=True)
                    if ind_row == 0:
                        df_graph = df_temp.copy()
                    else:
                        df_graph = pd.concat([df_graph, df_temp], ignore_index=True)
                    del df_temp
                #del df_topic_documents_hdbscan
                del df_topic_documents
                del df_topic_info
                del df_semtiment

                df_graph_temp = df_graph[['topic', 'setiment_scale']].groupby(by=['topic']).mean().reset_index()
                df_graph['setiment_scale_mean'] = 0
                for _, row_graph_temp in df_graph_temp.iterrows():
                    df_graph.loc[df_graph['topic'] == row_graph_temp['topic'], 'setiment_scale_mean'] = row_graph_temp['setiment_scale']
                del df_graph_temp

                if keep_documents_no_topic:
                    file_name = self.file_path + prefix + 'graph_data_with_no_topic.csv'
                else:
                    file_name = self.file_path + prefix + 'graph_data.csv'
                df_graph.to_csv(file_name, header=True, index=False, encoding='utf-8')
                del df_graph

if __name__ == '__main__':
    pass