#Preprocessing
#!pip install stop-words
#!python -m nltk.downloader punkt
#!python -m nltk.downloader averaged_perceptron_tagger

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re, string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import pkg_resources

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer



###################################
#### sentence level preprocess ####
###################################

# lowercase + base filter
# some basic normalization

def f_base(s):
  # norm 1: remove leading and trailing white spaces
  s = s.strip()
  # norm 2: lower case
  s = s.lower()
  # norm 3.1: remove patient id and date
  s = ' '.join(s.split(' ')[2:])
  # norm 3.2: remove radiology information
  s = re.sub(r'(jacqueline).*', '',s)
  s = re.sub(r'(gmc).*', '',s)
  s = re.sub(r'(radiology).*', '',s)
  s = re.sub(r'(consultant).*', '',s)
  s = s.replace('************************addendum start************************', '')
  s = s.replace('************************addendum end************************', '')
  s = re.sub(r'(dr ).*', '',s)
  s = re.sub(r'(ext: ).*', '',s)
  s = re.sub(r'(neil).*', '',s)
  s = re.sub(r'(shezad).*', '',s)
  s = re.sub(r'(neil).*', '',s)
  s = re.sub(r'(jacqueline).*', '',s)
  s = re.sub(r'(sattar).*', '',s)
  s = re.sub(r'(eeke).*', '',s)
  s = re.sub(r'(mr1).*', '',s)
  s = re.sub(r'(jennifer).*', '',s)
  # norm 4: remove break line
  s = s.replace('\\.br\\', '.')
  # norma 5: letter repetition (if more than 2)
  s = re.sub(r'([a-z])\1{2,}', r'\1', s)
  # norm 6: translate / to or
  s = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])', ' or ', s)
  s = re.sub('and/or', 'or', s)
  # norm 7: non-word repetition (if more than 1)
  s = re.sub(r'([\W+])\1{1,}', r'\1', s)
  # norm 8: remove html tags/markups
  s = re.compile('<.*?>').sub('', s)
  # norm 9: noise text - remove duplicate phrases
  s = s.replace('mri internal auditory meatus both','')
  s = s.replace('mri iam with contrast both','')
  s = s.replace('mri internal auditory','')
  s = s.replace('mri iams','')
  s = s.replace('findings;','')
  s = s.replace('findings:','')
  s = s.replace('indication:','')
  s = s.replace('comment;','')
  s = s.replace('comment:','')
  s = s.replace('erratum:','')
  s = s.replace('clinical indication:','')
  s = s.replace('conclusion:','')
  s = s.replace('conclusions:','')
  s = s.replace('common;','')
  s = s.replace('clinical history:','')
  s = s.replace('impression:','')
  s = s.replace('clinical details.','')
  s = s.replace('mri findings.','')
  s = s.strip()
  s = s.replace('(',' ')
  s = s.replace(')',' ')
  # norm 10: split the string using whitespace as the delimiter
  #          this removes all whitespace between words
  s = s.split()
  s = ' '.join(s)
  # norm 11: matches multiple whitespace
  s = re.sub(r'\s+',' ', s)
  # norm 12: remove leading and trailing white spaces
  s = s.strip()
  # norm 13: remove elipsoidal
  s = s.replace("..", ".")

  return s


snowballStemmer = SnowballStemmer('english')
wl = WordNetLemmatizer()

def f_base2(text):
  text = f_base(text)
  text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  #Replace punctuation with space.
  text = re.sub('\s+', ' ', text)  #Remove extra space and tabs
  text= re.sub(r'[^\w\s]', '', str(text).lower().strip())
  text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
  return text

def stopword(string):
  a = [i for i in string.split() if i not in stopwords.words('english')]
  return ' '.join(a)
  
def finalpreprocess(string):
  transformed_string = f_base2(string)
  transformed_string = stopword(transformed_string)
  transformed_string = lemmatizer(transformed_string)
  #return self.lemmatizer(self.stopword(self.preprocess(string)))
  return transformed_string


def lemmatizer(string):
  word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
  a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
  return " ".join(a)

def get_wordnet_pos(tag):
  if tag.startswith('J'):
      return wordnet.ADJ
  elif tag.startswith('V'):
      return wordnet.VERB
  elif tag.startswith('N'):
      return wordnet.NOUN
  elif tag.startswith('R'):
      return wordnet.ADV
  else:
      return wordnet.NOUN
  



###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct filtered out
    """
    #text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text) 
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list
en_stop = get_stop_words('en')


def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in en_stop]




def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = finalpreprocess(rw)
    #s = f_base2(rw)
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    #w_list = f_punct(w_list)
    #w_list = f_noun(w_list)
    #w_list = f_stem(w_list)
    #w_list = f_stopw(w_list)

    return w_list

def pre_process(docs, y_labels):
    """
    Preprocess the data
    Transform the output labels to numeric
    """

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
   

    samp = np.random.choice(n_docs, n_docs)
    for i in range(0, n_docs):
        sentence = preprocess_sent(docs[i])
        token_list = preprocess_word(sentence)
        if token_list:
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')

    le = LabelEncoder()
    y_labels = le.fit_transform(y_labels)

    print('Preprocessing raw texts. Done!')


    return sentences, token_lists, y_labels
