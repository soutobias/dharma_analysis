import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

class Similarity(object):

    def __init__(self):
        pass

    def calculate_metrics_npy(self, embeddings):
        metric_columns = ['document_1', 'document_2', 'pearson_corr', 'spearman_corr', 'kendalltau_corr', 'cosine_similarity', 'euclidean_distance', 'manhattan_distance']
        df_metrics = pd.DataFrame(columns=metric_columns)

        for i in range(0, embeddings.shape[0]):
            x = embeddings[i]
            for j in range(0, embeddings.shape[0]):
                y = embeddings[j]
                
                # calculate Pearson's correlation
                pearson_corr, _ = pearsonr(x, y)

                # calculate Spearman's correlation
                spearman_corr, _ = spearmanr(x, y)

                # calculate Pearson’s correlation
                kendalltau_corr, _ = kendalltau(x, y)

                # calculate Cosine Similarity
                # reshape the vectors x and y using .reshape(1, -1) to compute the cosine similarity for a single sample
                cos_sim = cosine_similarity(x.reshape(1,-1), y.reshape(1,-1)).flatten()[0]

                # calculate Euclidean Distance
                # Compared to the Cosine and Jaccard similarity, Euclidean distance is not used very often in the context of NLP applications
                dst_euclidean = distance.euclidean(x, y)

                # calculate Manhattan Distance
                dst_cityblock = distance.cityblock(x, y)
        
                new_metric = {}
                new_metric['document_1'] = i
                new_metric['document_2'] = j
                new_metric['pearson_corr'] = round(pearson_corr, 6)
                new_metric['spearman_corr'] = round(spearman_corr, 6)
                new_metric['kendalltau_corr'] = round(kendalltau_corr, 6)
                new_metric['cosine_similarity'] = round(cos_sim, 6)
                new_metric['euclidean_distance'] = round(dst_euclidean, 6)
                new_metric['manhattan_distance'] = round(dst_cityblock, 6)
                df_metrics = df_metrics.append(new_metric, ignore_index=True)
        return df_metrics

    def calculate_metrics_npz(self, embeddings):
        metric_columns = ['document_1', 'document_2', 'pearson_corr', 'spearman_corr', 'kendalltau_corr', 'cosine_similarity', 'euclidean_distance', 'manhattan_distance']
        df_metrics = pd.DataFrame(columns=metric_columns)

        for i in range(0, embeddings.shape[0]):
            x = embeddings[i]
            for j in range(0, embeddings.shape[0]):
                y = embeddings[j]
                
                # calculate Pearson's correlation
                pearson_corr, _ = pearsonr(x.toarray()[0], y.toarray()[0])

                # calculate Spearman's correlation
                spearman_corr, _ = spearmanr(x.toarray()[0], y.toarray()[0])

                # calculate Pearson’s correlation
                kendalltau_corr, _ = kendalltau(x.toarray()[0], y.toarray()[0])

                # calculate Cosine Similarity
                # reshape the vectors x and y using .reshape(1, -1) to compute the cosine similarity for a single sample
                cos_sim = cosine_similarity(x, y).flatten()[0]

                # calculate Euclidean Distance
                # Compared to the Cosine and Jaccard similarity, Euclidean distance is not used very often in the context of NLP applications
                dst_euclidean = distance.euclidean(x.toarray()[0], y.toarray()[0])

                # calculate Manhattan Distance
                dst_cityblock = distance.cityblock(x.toarray()[0], y.toarray()[0])
        
                new_metric = {}
                new_metric['document_1'] = i
                new_metric['document_2'] = j
                new_metric['pearson_corr'] = round(pearson_corr, 6)
                new_metric['spearman_corr'] = round(spearman_corr, 6)
                new_metric['kendalltau_corr'] = round(kendalltau_corr, 6)
                new_metric['cosine_similarity'] = round(cos_sim, 6)
                new_metric['euclidean_distance'] = round(dst_euclidean, 6)
                new_metric['manhattan_distance'] = round(dst_cityblock, 6)
                df_metrics = df_metrics.append(new_metric, ignore_index=True)
        return df_metrics

if __name__ == '__main__':
    pass