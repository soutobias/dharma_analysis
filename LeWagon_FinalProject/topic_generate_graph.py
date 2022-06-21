import pandas as pd
import numpy as np
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from LeWagon_FinalProject.sentiment import Sentiment

class GenerateGraphData(object):

    def __init__(self, file_path):
        ''' file_path: path to save datasets created '''

        self.file_path = file_path

    def generate_topic_info(self, bert_model, file_name):
        df_topic_info = bert_model.get_topic_info()
        df_topic_info.to_csv(file_name, header=True, index=False, encoding='utf-8')
        del bert_model
        return df_topic_info

    def generate_terms(self, bert_model, file_name):
        topics = bert_model.get_topics()
        number_of_topics = len(topics)-1
        num_of_terms = len(topics[0])

        topic_columns = ['topic', 'term', 'weight']

        df_topics = pd.DataFrame(columns=topic_columns)
        for i in range(0,number_of_topics):
            for j in range(num_of_terms):
                new_topic = {}
                new_topic['topic'] = bert_model.topic_names[i]
                new_topic['term'] = topics[i][j][0]
                new_topic['weight'] = round(topics[i][j][1],6)
                df_topics = df_topics.append(new_topic, ignore_index=True)
        df_topics.to_csv(file_name, header=True, index=False, encoding='utf-8')
        del bert_model
        del df_topics

    def correlation_matrix_to_df(self, df_corr):
        list_done = []
        lits_item1 = []
        lits_item2 = []
        list_corr = []

        for k in range(1,df_corr.shape[1]):
            for i, j in df_corr.iterrows():
                #if (df_corr.columns[k] != j[0]) and (j[0] not in list_done):
                #if (j[0] not in list_done):
                lits_item1.append(df_corr.columns[k])
                lits_item2.append(j[0])
                list_corr.append(j[k])
            list_done.append(df_corr.columns[k])

        corr_dict = {'topic1': lits_item1,
                    'topic2': lits_item2,
                    'similarity': list_corr}
        df_res = pd.DataFrame(corr_dict)
        df_res = df_res.sort_values(by='similarity', ascending=False).copy()
        df_res.reset_index(inplace=True,drop=True)
        del df_corr
        return df_res.copy() 

    def generate_topic_similarity(self, bert_model, file_name):
        corr_matrix = bert_model.topic_sim_matrix

        topics = bert_model.get_topics()
        number_of_topics = len(topics)-1

        topic_columns = ['topic']
        for i in range(-1,number_of_topics):
            topic_columns.append(bert_model.topic_names[i])

        df_similarity = pd.DataFrame(columns=topic_columns)
        for i in range(-1,number_of_topics):
            new_topic = {}
            new_topic['topic'] = bert_model.topic_names[i]
            for j in range(-1,number_of_topics):
                new_topic[bert_model.topic_names[j]] = round(corr_matrix[i,j],6)
            df_similarity = df_similarity.append(new_topic, ignore_index=True)
            
        df_topic_similarity = self.correlation_matrix_to_df(df_similarity)
        df_topic_similarity.to_csv(file_name, header=True, index=False, encoding='utf-8')
        del df_similarity
        del corr_matrix
        del bert_model
        del df_topic_similarity

    def get_topic_documents(self, cluster_id, condensed_tree):
        result_points = np.array([])
        result_points_val = np.array([])
        
        #assert cluster_id > -1, "The topic's label should be greater than -1!"
        
        if cluster_id <= -1:
            return result_points.astype(np.int64), result_points_val.astype(np.float64)
            
        raw_tree = condensed_tree._raw_tree
        
        # Just the cluster elements of the tree, excluding singleton points
        cluster_tree = raw_tree[raw_tree['child_size'] > 1]
        
        # Get the leaf cluster nodes under the cluster we are considering
        leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
        
        # Now collect up the last remaining points of each leaf cluster (the heart of the leaf) 
        for leaf in leaves:
            #max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
            #points = raw_tree['child'][(raw_tree['parent'] == leaf) & (raw_tree['lambda_val'] == max_lambda)]
            #points_val = raw_tree['lambda_val'][(raw_tree['parent'] == leaf) & (raw_tree['lambda_val'] == max_lambda)]
            points = raw_tree['child'][(raw_tree['parent'] == leaf)]
            points_val = raw_tree['lambda_val'][(raw_tree['parent'] == leaf)]
            result_points = np.hstack((result_points, points))
            result_points_val = np.hstack((result_points_val, points_val))   
        return result_points.astype(np.int64), result_points_val.astype(np.float64)

    def generate_topic_documents(self, bert_model, file_name):
        try:
            clusterer = bert_model.hdbscan_model
            tree = clusterer.condensed_tree_
            clusters = tree._select_clusters()

            number_of_topics = len(clusters)

            relevant_columns = ['topic', 'document', 'lambda_val']
            df_rel_docs = pd.DataFrame(columns=relevant_columns)

            for i in range(-1, number_of_topics):
                rel_docs, lambda_vals = self.get_topic_documents(clusters[i], tree)
                topic_name = bert_model.topic_names[i]
                for j in range(0, len(rel_docs)):
                    new_doc_rel = {}
                    new_doc_rel['topic'] = topic_name
                    new_doc_rel['document'] = rel_docs[j]
                    new_doc_rel['lambda_val'] = round(lambda_vals[j],6)
                    df_rel_docs = df_rel_docs.append(new_doc_rel, ignore_index=True)
            df_rel_docs.to_csv(file_name, header=True, index=False, encoding='utf-8')
            del bert_model
            del df_rel_docs
        except:
            pass

    def generate_topic_documents_hdbscan(self, bert_model, file_name):
        clusterer = bert_model.hdbscan_model

        doc_topic_columns = ['document', 'topic', 'probabilities']
        df_doc_topic = pd.DataFrame(columns=doc_topic_columns)

        for i, _ in enumerate(clusterer.labels_):
            new_doc_topic = {}
            new_doc_topic['document'] = i
            new_doc_topic['topic'] = clusterer.labels_[i]
            new_doc_topic['probabilities'] = clusterer.probabilities_[i]
            df_doc_topic = df_doc_topic.append(new_doc_topic, ignore_index=True)
        df_doc_topic.to_csv(file_name, header=True, index=False, encoding='utf-8')
        del bert_model
        del df_doc_topic
        
    def generate_documents_similarity(self, bert_model, docs, file_name):
        emb_model = bert_model.embedding_model
        # Create documents embeddings
        embeddings = emb_model.embedding_model.encode(docs)
        doc_sim_matrix = cosine_similarity(embeddings, embeddings)
        np.save(file_name, doc_sim_matrix)
        del bert_model
        del doc_sim_matrix

    def generate_documents_sentiment(self, df_news, file_name):
        contet_size = 1200        
        df_news = df_news.copy()
        sentiment = Sentiment(df_news['content'])
        do_it = True
        while do_it:
            try:
                sentiment.sentiment_analisys(contet_size)
                do_it = False
            except:
                contet_size -= 100
                do_it = True
        sentiment.pred.to_csv(file_name, header=True, index=True, encoding='utf-8')
        del sentiment

    def create_year_month_datasets(self, base_dataset_path, min_topic_size=10):
        ''' Based on the base dataset, creates the below datasets
            base_dataset_path: path of dataset to split by year and month (e.g. ../raw_data/proj_final/political_dataset.csv)
                mandatory columns of base dataset: [title, year, month, content]
            list_of_years_months.csv: dataset with the split months and years found in base dataset
            BERTopic_Info.csv: topic information for each year and month
            BERTopic_TopicSimilarity.npy: similarity of topics for each year and month
            BERTopic_DocumentsSimilarity.npy: similarity of documents for each year and month
            HDBSCAN_TopicDocuments.csv: documents of each topic found
        '''

        df = pd.read_csv(base_dataset_path)
        df['title'].fillna('no title', inplace = True)
        df = df.sort_values(by=['year', 'month'], ascending=True).reset_index(drop=True)
        df_split = df[['id', 'year', 'month']].groupby(['year', 'month']).count().reset_index()

        list_prefix = []
        for row_id, row in df_split.iterrows():
            print(str(row['year']) + ' - ' + str(row['month']))

            prefix = f'{str(int(row["year"]))}_{str(int(row["month"]))}_'
            list_prefix.append(prefix)

            #if ((row['year'] >= 2020) and (row['month'] >= 8)) or (row['year'] > 2016):
            df_temp = df[(df['year'] == df_split['year'].iloc[row_id]) & (df['month'] == df_split['month'].iloc[row_id])].copy()
            df_temp = df_temp.sort_values(by=['year', 'month'], ascending=True).reset_index(drop=True)
            docs = df_temp['content'].values
            
            file_name = self.file_path + prefix + 'dataset.csv' 
            df_temp.to_csv(file_name, header=True, index=True, encoding='utf-8')
            if df_temp.shape[0] > min_topic_size:
                sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')
                topic_model = BERTopic(min_topic_size=min_topic_size, language='english', calculate_probabilities=True, n_gram_range=(2,2), embedding_model=sentence_model)
                topics, probs = topic_model.fit_transform(docs)              

                df_topic_docs = pd.DataFrame(data={'document_id': df_temp.index.values, 'topic': topics})
                file_name = self.file_path + prefix + 'BERTopic_TopicDocuments.csv' 
                df_topic_docs.to_csv(file_name, header=True, index=False, encoding='utf-8')

                file_name = self.file_path + prefix + 'BERTopic_TopicDocumentsProbs.npy' 
                np.save(file_name, probs)

                del df_topic_docs
                print('fit_transform done ...')                
                
                file_name = self.file_path + prefix + 'BERTopic_model_2_2_raw_content' 
                topic_model.save(file_name)

                file_name = self.file_path + prefix + 'BERTopic_Info.csv'
                df_topic_info = self.generate_topic_info(topic_model, file_name)

                if df_topic_info.shape[0] > 20:
                    new_topics, new_probs = topic_model.reduce_topics(docs, topics, probabilities=probs, nr_topics=20)
                    df_topic_docs = pd.DataFrame(data={'document_id': df_temp.index.values, 'topic': new_topics})
                    file_name = self.file_path + prefix + 'BERTopic_TopicDocuments_reduction.csv'
                    df_topic_docs.to_csv(file_name, header=True, index=False, encoding='utf-8')

                    file_name = self.file_path + prefix + 'BERTopic_TopicDocumentsProbs_reduction.npy' 
                    np.save(file_name, new_probs)
                    del df_topic_docs

                    file_name = self.file_path + prefix + 'BERTopic_Info_reduction.csv'
                    df_topic_info = self.generate_topic_info(topic_model, file_name)
                    print('topics reduction done ...')

                if df_topic_info.shape[0] > 1:
                    file_name = self.file_path + prefix + 'sentiment.csv'
                    self.generate_documents_sentiment(df_temp, file_name)
                    print('documents_sentiment done ...')

                    file_name = self.file_path + prefix + 'BERTopic_DocumentsSimilarity.npy'
                    self.generate_documents_similarity(topic_model, docs, file_name)
                    print('documents_similarity done ...')

                    file_name = self.file_path + prefix + 'BERTopic_Terms.csv'  
                    self.generate_terms(topic_model, file_name)
                    
                    #file_name = self.file_path + prefix + 'BERTopic_TopicSimilarity.csv'
                    #self.generate_topic_similarity(topic_model, file_name)
                    file_name = self.file_path + prefix + 'BERTopic_TopicSimilarity.npy'
                    np.save(file_name, topic_model.topic_sim_matrix)
                    
                    #file_name = self.file_path + prefix + 'BERTopic_TopicDocuments.csv'  
                    #self.generate_topic_documents(topic_model, file_name)
                    
                    #file_name = self.file_path + prefix + 'HDBSCAN_TopicDocuments.csv'  
                    #self.generate_topic_documents_hdbscan(topic_model, file_name)
                
                del topic_model
                del sentence_model
            del df_temp
            del docs
            
        df_prefix = pd.DataFrame(list_prefix, columns=['year_month'])
        file_name = self.file_path + 'list_of_years_months.csv' 
        df_prefix.to_csv(file_name, header=True, index=True, encoding='utf-8')
        del df_prefix

if __name__ == '__main__':
    pass