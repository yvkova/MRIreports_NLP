# -*- coding: utf-8 -*-

from nltk import ngrams
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import pad_both_ends

import gensim
from gensim.downloader import load
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from gensim.models import word2vec

import itertools, collections
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MSc Project/'

def frequency_with_word(term, top_n, token_lists):
  frequency = {}
  for report in token_lists:
    if (term) in report:
      for word in report:
        if word in frequency:
          frequency.update({word: (frequency[word]+1)}) 
        else:
          frequency[word]=1
  result = dict(sorted(frequency.items(), key=lambda x:x[1], reverse=True))
  return dict(itertools.islice(result.items(), top_n))

def get_word_pairs(token_lists):
  tokens = list(flatten(pad_both_ends(sent, n=2) for sent in token_lists))
  counted_2= collections.Counter(ngrams(tokens,2))
  word_pairs = pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
  return word_pairs

def get_frequent_word_pair_with(terms, token_lists, top_n=20):
  word_pair = get_word_pairs(token_lists)

  if terms is not None:
    res = word_pair.pairs.apply(lambda x: any(val in x for val in terms))
    word_pair = word_pair[res]
  
  res = word_pair.pairs.apply(lambda x: any((val not in x for val in ['<s>'])))
  word_pair = word_pair[res]
  res = word_pairs.pairs.apply(lambda x: any((val not in x for val in ['</s>'])))
  return word_pair[res][0:top_n]

def tsne_plot(model, name):

  "Creates and TSNE model and plots it"
  labels = []
  tokens = []

  for word in model.wv.vocab:
      tokens.append(model[word])
      labels.append(word)
  
  tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
  new_values = tsne_model.fit_transform(tokens)

  x = []
  y = []
  for value in new_values:
      x.append(value[0])
      y.append(value[1])
      
  plt.figure(figsize=(20, 20)) 
  for i in range(len(x)):
      plt.scatter(x[i],y[i])
      plt.annotate(labels[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
  dr = DRIVE_LOCATION +'/files/eda_results/TSNE/{}'.format(name)
  plt.savefig(dr)
  plt.show()

def get_overall_word_frequencies(token_lists):
  tokens = []
  for sent in token_lists:
    for word in sent:
      tokens.append(word)

  df = pd.DataFrame(tokens)
  df = df[0].value_counts()

  df = df[:20,]
  plt.figure(figsize=(10,5))
  sns.barplot(df.values, df.index, alpha=0.8)
  plt.title('Top Words Overall')
  plt.ylabel('Word from Clinical Report', fontsize=12)
  plt.xlabel('Count of Words', fontsize=12)
  #plt.show()
  dr = DRIVE_LOCATION +'/files/eda_results/{}'.format('top_words_overall')
  plt.savefig(dr)

# To get synonyms of key words
def apply_bioWordVec_to_tokens(token_lists):
  tokens = []
  for sent in token_lists:
    for word in sent:
      tokens.append(word)

  df = pd.DataFrame(tokens)
  df = df[0].value_counts()
  labels = df.index
  print('Getting vector representations for BioWordVec ...')
  model = KeyedVectors.load_word2vec_format(DRIVE_LOCATION +'/files/BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
  print('Getting vector representations for BioWordVec. Done!')

  lbs = []
  tokens = []
  for w in labels:
    if w in model.vocab:
      tokens.append(model.wv.get_vector(w))
      lbs.append(w)

  return model, lbs, tokens

def get_synonym(model, words_of_interest, labels ,tokens_wv, top_n=50):
  similarities = {}
  similar_df = pd.DataFrame(labels , columns=['words'])
  for word in words_of_interest:
    word_of_interest_index = 0
    for i in range(0,len(labels)):
      if word == labels[i]:
        word_of_interest_index = i
    similarities[word] = model.cosine_similarities(tokens_wv[word_of_interest_index], tokens_wv)
    similar_df[word] = similarities[word]
  return similar_df

#Let's create a reusable function to explore the word count through preprocessing
def word_level_stats(data, column_name):
    words = []
    for document in data[column_name]:
        words.extend(document.split())

    total_words = len(words)
    print("Total Number of words:", total_words)

    unique_words = len(set(words))
    print("Total Number of unique words aka vocabulary size:", unique_words)

    average_count = int(total_words/unique_words)
    print("Average count:",average_count)
    
    #return words

def word_level_stats(data):
    words = []
    for document in data:
        words.extend(document.split())

    total_words = len(words)
    print("Total Number of words:", total_words)

    unique_words = len(set(words))
    print("Total Number of unique words aka vocabulary size:", unique_words)

    average_count = int(total_words/unique_words)
    print("Average count:",average_count)
    
    return words

def get_frequent_words(word_list, n):
    total_count = len(word_list)
    count_dict = dict(Counter(word_list))
    count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse = True)
    n_frequent = count_dict[:n]
    
    for word, count in n_frequent:
        print("word:{0}, percentage:{1}".format(word,round(count/total_count*100,2)))
