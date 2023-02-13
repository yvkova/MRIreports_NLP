from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from gensim import corpora
import gensim
import numpy as np

from datetime import datetime

from vec_embeddings import VecEmbeddings


class USLModel:

  def __init__(self, sentences, token_lists, labels, vec_method, vec={}):
    self.cluster_method = 'kmeans'
    self.k = 2
    self.vec_method = vec_method
    self.sentences = sentences
    self.token_lists = token_lists
    self.labels = labels
    self.vec_embeddings = VecEmbeddings(sentences, token_lists, labels, vec_method)
    self.vec_method = vec_method
    self.vec = vec
    self.cluster_model = None
    self.cluster_label = None
    #DBSCAN
    self.eps = 0.2
    self.min_samples = 5
    self.id = self.vec_method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

  
  def cluster(self, cluster_method, vec_method, k=2):
    self.k=k
    if vec_method is None:
      vec_method = self.vec_method
    else:
      self.vec_method = vec_method
    if cluster_method is None:
      cluster_method = self.cluster_method
    else:
      self.cluster_method = cluster_method

    if ((self.vec == {}) | (vec_method not in self.vec) ) :
      print('Vector embedding does not exist.')
      self.vec[vec_method] = self.vec_embeddings.vectorize(vec_method)
    else:
      print('Vector embedding already exists, skipping vector creation.')
    
    if (cluster_method == 'kmeans'):
      self.cluster_model = KMeans(self.k)
      self.cluster_model.fit(self.vec[vec_method])
      print('KMeans Clustering embeddings. Done!')
    
    elif (cluster_method == 'gmm'):
      self.cluster_model = GMM(n_components=self.k, n_init=10)
      self.cluster_model.fit(self.vec[vec_method])
      self.cluster_label = self.cluster_model.predict(self.vec[vec_method])
      print('GMM Clustering embeddings. Done!')

    elif (cluster_method == 'birch'):
      self.cluster_model = Birch(n_clusters=self.k)
      self.cluster_model.fit(self.vec[vec_method])
      self.cluster_label = self.cluster_model.predict(self.vec[vec_method])
      print('Birch Clustering embeddings. Done!')

    elif (cluster_method == 'dbscan'):
      self.cluster_model = DBSCAN(eps=self.eps, min_samples= self.min_samples)
      self.cluster_model.fit(self.vec[vec_method])
      self.cluster_label = self.cluster_model.predict(self.vec[vec_method])
      print('DBSCAN Clustering embeddings. Done!')

    elif (cluster_method == 'gmm2'):
      self.cluster_model = GMM(n_components=self.k, n_init=10)
      self.cluster_model.fit(self.vec[vec_method])
      self.cluster_label = self.cluster_model.predict_proba(self.vec[vec_method])
      print('GMM Clustering embeddings. Done!')
