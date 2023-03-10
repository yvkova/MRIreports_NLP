from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import os

from usl_model import USLModel

DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MRI Project/'


def print_metrics(y_labels, y_predictions, label=None): 
    accuracy = accuracy_score(y_labels, y_predictions)
    precision = precision_score(y_labels, y_predictions)
    recall = recall_score(y_labels, y_predictions)
    auc = roc_auc_score(y_labels, y_predictions)
    print("Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, AUC: {:.2}".format(accuracy, precision, recall, auc))
    print('Confusion Matrix: \n',confusion_matrix(y_labels, y_predictions, labels=label)) 
    print("\n Balanced Accuracy", balanced_accuracy_score(y_labels, 
                                                  y_predictions))

class EvaluationMetrics:

  def __init__(self, model:USLModel, no_aug: False):
    self.model = model
    self.no_aug = no_aug


  def get_match_score(self, cluster_index, gt_index, labels):
    return len(labels[(labels['clusters']==cluster_index) & (labels['ground_truth']==gt_index)])


  def get_label_df_exp2(self, model):
    """
    Ensure that the clusters are assigned the correct label
    0 - Negative for tumour
    1 - Postive for tumour
    2 - Other
    :param model: USLModel object
    :return: label dataframe containing cluser labels and ground truth
    """
    lbs = None
    if(model.cluster_method == 'gmm'):
      lbs = model.cluster_label
    else:
      lbs = model.cluster_model.labels_
    
    labels = pd.DataFrame(lbs , columns=['clusters'])
    labels['ground_truth']= model.labels

    if(self.no_aug == True):
      labels = labels[0:6117]

    arr = [0,0,0,0,0,0]
    index_pair = [(0,0), (1,0), (2, 0), (0,1), (1,1), (2,1)]

    for i in range(0,len(index_pair)):
      arr[i] = self.get_match_score(index_pair[i][0], index_pair[i][1], labels)

    c_0 = index_pair[np.argmax(arr[0:3], axis=-1)]
    c_1 = index_pair[np.argmax(arr[3:6], axis=-1)+3]

    if(c_0[0] != c_0[1]): 
      labels.clusters[(labels['clusters']==c_0[0])] = 3

      if((c_1[0] != c_1[1]) & (c_0[0] == c_1[1])):
        labels.clusters[(labels['clusters']==c_1[0])] = c_1[1]
        labels.clusters[(labels['clusters']==c_0[1])] = c_1[0]
        labels.clusters[(labels['clusters']==3)] = c_0[1]

    return labels

  def get_label_df_new(self, model):
    """
    Ensure that the clusters are assigned the correct label
    0 - Negative for tumour
    1 - Postive for tumour
    :param model: USLModel object
    :return: label dataframe containing cluser labels and ground truth
    """
    lbs = None
    if(model.cluster_method == 'gmm'):
      lbs = model.cluster_label
    else:
      lbs = model.cluster_model.labels_
    
    labels = pd.DataFrame(lbs , columns=['clusters'])
    labels['ground_truth']= model.labels

    arr = [0,0,0,0]
    index_pair = [(0,0), (0,1), (1,0), (1,1)]

    for i in range(0,len(index_pair)):
      arr[i] = self.get_match_score(index_pair[i][0], index_pair[i][1], labels)

    c = index_pair[np.argmax(arr, axis=-1)]
    arr[np.argmax(arr, axis=-1)] = 0
    c_2 = index_pair[np.argmax(arr, axis=-1)]

    if((c[0] != c[1])): #& (c_2[0] != c_2[1])):
      labels.clusters[(labels['clusters']==c[0])] = 3
      labels.clusters[(labels['clusters']==c[1])] = c[0]
      labels.clusters[(labels['clusters']==3)] = c[1]
    
    if(self.no_aug == True):
      labels = labels[0:6117]

    return labels

  def get_metrics(self, model):
    """
    Get accuracy, precision, recall and AUC
    :param model: USLModel object
    :return: classification metrics
    """
    lbs = self.get_label_df_new(model)
    accuracy = accuracy_score(lbs['ground_truth'], lbs['clusters'])
    precision = precision_score(lbs['ground_truth'], lbs['clusters'], pos_label=1)
    recall = recall_score(lbs['ground_truth'], lbs['clusters'], pos_label=1)
    precision_neg = precision_score(lbs['ground_truth'], lbs['clusters'], pos_label=0)
    recall_neg = recall_score(lbs['ground_truth'], lbs['clusters'], pos_label=0)
    auc = roc_auc_score(lbs['ground_truth'], lbs['clusters'])
    f1= f1_score(lbs['ground_truth'], lbs['clusters'], pos_label=1)
    f1_neg= f1_score(lbs['ground_truth'], lbs['clusters'], pos_label=0)
    return accuracy, precision, recall, precision_neg, recall_neg, auc, f1, f1_neg

  
  def get_silhouette(self, model):
    """
    Get silhouette score from model
    :param model: USLModel object
    :return: silhouette score
    """
    if model.vec_method == 'LDA':
        return
    lbs = None
    if(model.cluster_method == 'gmm'):
      lbs = model.cluster_label
    else:
      lbs = model.cluster_model.labels_
    vec = model.vec[model.vec_method]

    if(self.no_aug == True):
      lbs = lbs[0:6117]
      vec = vec[0:6117]
    
    return silhouette_score(vec, lbs)

  def get_davies_bouldin_score(self, model):
    """
    Get davies bouldin score from model
    :param model: USLModel object
    :return: davies_bouldin_score
    """
    if model.vec_method == 'LDA':
        return
    lbs = None
    if(model.cluster_method == 'gmm'):
      lbs = model.cluster_label
    else:
      lbs = model.cluster_model.labels_
    vec = model.vec[model.vec_method]
    #if model.vec_method == 'TFIDF':
    #  vec = vec.toarray()

    if(self.no_aug == True):
      lbs = lbs[0:6117]
      vec = vec[0:6117]

    return davies_bouldin_score(vec, lbs)


  def get_fowlkes_mallows_score(self, model):
    """
    Get fowlkes mallows score from model
    :param model: USLModel object
    :return: fowlkes mallows score
    """
    if model.vec_method == 'LDA':
        return
    lbs = self.get_label_df_new(model)
    return fowlkes_mallows_score(lbs['ground_truth'], lbs['clusters'])

  def get_adjusted_rand_score(self, model):
    """
    Get adjusted rand score from model
    :param model: USLModel object
    :return: adjusted rand score
    """
    if model.vec_method == 'LDA':
        return
    lbs = self.get_label_df_new(model)
    return adjusted_rand_score(lbs['ground_truth'], lbs['clusters'])
    

  def get_evaluation_metrics(self, model):
    accuracy, precision, recall, precision_neg, recall_neg, auc, f1, f1_neg = self.get_metrics(model)

    return pd.DataFrame(data = [[model.vec_method, 
                                self.get_fowlkes_mallows_score(model), 
                                self.get_silhouette(model), 
                                self.get_adjusted_rand_score(model), 
                                self.get_davies_bouldin_score(model), 
                                accuracy,
                                precision,
                                recall,
                                precision_neg,
                                recall_neg,
                                auc,
                                f1,
                                f1_neg,
                                model.id]], 
                        columns=['Model', 
                            'FMS',
                            'SC',
                            'ARS', 
                            'DBS', 
                            'acc',
                            'prec',
                            'recall',
                            'prec_neg',
                            'recall_neg',
                            'auc',
                            'f1',
                            'f1_neg',
                            'Model ID'])

  def get_evaluation_metrics_exp2(self, model):
      return pd.DataFrame(data = [[model.vec_method, 
                                  self.get_silhouette(model), 
                                  self.get_davies_bouldin_score(model), 
                                  model.id]], 
                          columns=['Model', 
                              'SC',
                              'DBS', 
                              'Model ID'])

  def plot_proj(self, embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend()


  def visualize_original(self, model, experiment):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    if model.vec_method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.vec_method][0:6117])
    print('Calculating UMAP projection. Done!')
    self.plot_proj(vec_umap, model.cluster_model.labels_[0:6117])
    dr = DRIVE_LOCATION +'/files/results/{}/word_cloud/{}'.format(experiment, (model.cluster_method+'_'+model.id))
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_vis')

  def visualize(self, model, experiment):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    if model.vec_method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec = model.vec[model.vec_method]
    lbs = None
    if(experiment == 'experiment_2'):
      lbs = self.get_label_df_exp2(model)['clusters']
    else:
      lbs = self.get_label_df_new(model)['clusters']
    if(self.no_aug == True):
      lbs = lbs[0:6117]
      vec = vec[0:6117]
    vec_umap = reducer.fit_transform(vec)
    print('Calculating UMAP projection. Done!')
    self.plot_proj(vec_umap, lbs)
    dr = DRIVE_LOCATION +'/files/results/{}/word_cloud/{}'.format(experiment, (model.cluster_method+'_'+model.id))
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_vis')

  def get_wordcloud(self, model, token_lists, topic, experiment):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param experiment: experiment_1, experiment_1_corrected, experiment_2
    :param sentences: preprocessed sentences from docs
    """
    if model.vec_method == 'LDA':
        return
    print('Getting wordcloud for topic {} ...'.format(topic))
    lbs = None

    if(experiment == 'experiment_2'):
      lbs = self.get_label_df_exp2(model)['clusters']
    else:
      lbs = self.get_label_df_new(model)['clusters']
    if(self.no_aug == True):
      lbs = lbs[0:6117]
      token_lists = token_lists[0:6117]
    tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])

    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    dr = DRIVE_LOCATION +'/files/results/{}/word_cloud/{}'.format(experiment, (model.cluster_method+'_'+model.id))
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))

  # Helper functions to calculate optimal k

  def getSilhouetteScores(df, kmeans_per_k):
    return [silhouette_score(df, model.labels_)
                      for model in kmeans_per_k[1:]]

  def plotSilhouetteScores(silhouette_scores):
    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.show()
