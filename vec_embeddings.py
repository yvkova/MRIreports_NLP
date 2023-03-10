from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
from gensim.downloader import load
from gensim.models import KeyedVectors
import numpy as np

from datetime import datetime
from sentence_transformers import SentenceTransformer
from autoencoder import Autoencoder
from pca import DimensionalityReductionPCA
import sent2vec

DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MRI Project/'


class VecEmbeddings:

  def __init__(self, sentences, token_lists, labels, vec_method='TFIDF'):
    if vec_method not in {'TFIDF', 'Word2Vec', 'BioWordVec', 'BioSentVec','BERT','BLUEBERT', 'Bio_ClinicalBERT', 'SciBERT','PubMedBERT','BIOBERT', 'BIOBERT2', 'BIOBERT3', 'BIOBERT4' ,'LDA_BERT', 'LDA_Bio_ClinicalBERT', 'LDA_BIOBERT4'}:
      raise Exception('Invalid method!')

    self.sentences = sentences
    self.token_lists = token_lists
    self.labels = labels
    self.vec_method = vec_method
    self.dictionary = None
    self.corpus = None
    self.ldamodel = None
    self.gamma = 15  # parameter for reletive importance of lda
    self.k = 2
    self.AE = None

  def vectorize(self, method):

    # turn tokenized documents into a id <-> term dictionary
    self.dictionary = corpora.Dictionary(self.token_lists)
    # convert tokenized documents into a document-term matrix
    self.corpus = [self.dictionary.doc2bow(text) for text in self.token_lists]
    import tensorflow as tf
    if method == 'TFIDF':
      print('Getting vector representations for TF-IDF ...')
      tfidf = TfidfVectorizer()
      vec = tfidf.fit_transform(self.sentences)
      print('Getting vector representations for TF-IDF. Done!')
      dim_red_pca = DimensionalityReductionPCA(vec)
      print('Getting PCA vector representations for TF-IDF...')
      dim_red_pca._reduce()
      print('Getting PCA vector representations for TF-IDF. Done!')
      print(f'Shape before: {dim_red_pca.vec.shape[0]},{dim_red_pca.vec.shape[1]}')
      print(f'Shape after: {dim_red_pca.x_pca.shape[0]},{dim_red_pca.x_pca.shape[1]}')
      return dim_red_pca.x_pca
    
    elif method == 'LDA':
      print('Getting vector representations for LDA ...')
      if not self.ldamodel:
          self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                          passes=20)

      def get_vec_lda(model, corpus, k):
          """
          Get the LDA vector representation (probabilistic topic assignments for all documents)
          :return: vec_lda with dimension: (n_doc * n_topic)
          """
          n_doc = len(corpus)
          vec_lda = np.zeros((n_doc, k))
          for i in range(n_doc):
              # get the distribution for the i-th document in corpus
              for topic, prob in model.get_document_topics(corpus[i]):
                  vec_lda[i, topic] = prob

          return vec_lda

      vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
      print('Getting vector representations for LDA. Done!')
      return vec

    elif method == 'Word2Vec':
      print('Getting vector representations for Word2Vec ...')
      model = load('word2vec-google-news-300')

      def get_vect(word_tokens):
        return np.mean([model.wv.get_vector(w) for w in word_tokens if w in model.vocab], axis=0)
      vec = [get_vect(word_tokens) for word_tokens in self.token_lists]
      print('Getting vector representations for Word2Vec. Done!')
      return vec


    elif method == 'BioWordVec':
      print('Getting vector representations for BioWordVec ...')
      model = KeyedVectors.load_word2vec_format(DRIVE_LOCATION +'/files/BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
      def get_vect(word_tokens):
        return np.mean([model.wv.get_vector(w) for w in word_tokens if w in model.vocab], axis=0)
      vec = [get_vect(word_tokens) for word_tokens in self.token_lists]
      print('Getting vector representations for BioWordVec. Done!')
      return vec

    elif method == 'BioSentVec':
      print('Getting vector representations for BioSentVec ...')
      model = sent2vec.Sent2vecModel()
      model.load_model(DRIVE_LOCATION +'/files/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
      vec = model.embed_sentences(self.sentences)
      print('Getting vector representations for BioSentVec. Done!')
      return vec
    
    elif method == 'BERT':
      print('Getting vector representations for BERT ...')
      model = SentenceTransformer('bert-base-nli-max-tokens')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BERT. Done!')
      return vec

    elif method == 'BLUEBERT':
      print('Getting vector representations for BLUEBERT ...')
      model = SentenceTransformer('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BLUEBERT. Done!')
      return vec
    
    elif method == 'Bio_ClinicalBERT':
      print('Getting vector representations for Bio_ClinicalBERT ...')
      model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for Bio_ClinicalBERT. Done!')
      return vec
    
    elif method == 'SciBERT':
      print('Getting vector representations for SciBERT ...')
      model = SentenceTransformer('allenai/scibert_scivocab_uncased')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for SciBERT. Done!')
      return vec
    
    elif method == 'PubMedBERT':
      print('Getting vector representations for PubMedBERT ...')
      model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for PubMedBERT. Done!')
      return vec

    elif method == 'BIOBERT':
      print('Getting vector representations for BIOBERT ...')
      model = SentenceTransformer('dmis-lab/biobert-v1.1')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BIOBERT. Done!')
      return vec
    
    elif method == 'BIOBERT2':
      print('Getting vector representations for BIOBERT2 ...')
      model = SentenceTransformer('dmis-lab/biobert-large-cased-v1.1')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BIOBERT2. Done!')
      return vec

    elif method == 'BIOBERT3':
      print('Getting vector representations for BIOBERT3 ...')
      model = SentenceTransformer('dmis-lab/biosyn-biobert-bc2gn')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BIOBERT3. Done!')
      return vec
    
    elif method == 'BIOBERT4':
      print('Getting vector representations for BIOBERT4 ...')
      model = SentenceTransformer('dmis-lab/biosyn-sapbert-bc5cdr-disease')
      vec = np.array(model.encode(self.sentences, show_progress_bar=True))
      print('Getting vector representations for BIOBERT4. Done!')
      return vec
    
    elif method == 'LDA_Bio_ClinicalBERT':
      vec_lda = self.vectorize(method='LDA')
      vec_bert = self.vectorize(method='Bio_ClinicalBERT')
      vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
      #self.vec['LDA_BERT_FULL'] = vec_ldabert
      if not self.AE:
          self.AE = Autoencoder()
          print('Fitting Autoencoder ...')
          self.AE.fit(vec_ldabert)
          print('Fitting Autoencoder Done!')
      vec = self.AE.encoder.predict(vec_ldabert)
      return vec
    
    elif method == 'LDA_BIOBERT4':
      vec_lda = self.vectorize(method='LDA')
      vec_bert = self.vectorize(method='BIOBERT4')
      vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
      #self.vec['LDA_BERT_FULL'] = vec_ldabert
      if not self.AE:
          self.AE = Autoencoder()
          print('Fitting Autoencoder ...')
          self.AE.fit(vec_ldabert)
          print('Fitting Autoencoder Done!')
      vec = self.AE.encoder.predict(vec_ldabert)
      return vec

    else:
      vec_lda = self.vectorize(method='LDA')
      vec_bert = self.vectorize(method='BERT')
      vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
      #self.vec['LDA_BERT_FULL'] = vec_ldabert
      if not self.AE:
          self.AE = Autoencoder()
          print('Fitting Autoencoder ...')
          self.AE.fit(vec_ldabert)
          print('Fitting Autoencoder Done!')
      vec = self.AE.encoder.predict(vec_ldabert)
      return vec
