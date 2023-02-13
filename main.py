# The code was inspired by https://github.com/Stveshawn/contextual_topic_identification/blob/master/

from preprocess import *
from vec_embeddings import VecEmbeddings
from usl_model import USLModel
from evaluation_metrics import *
from data_augmentation import augment
from eda import *
from utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

# Clustering on original dataset
def experiment_1():
  mri_report_df = read_data(DRIVE_LOCATION + "Data/MRIreports.xlsx")
  sentences, token_lists, y_labels = pre_process(mri_report_df.report_text, mri_report_df.tumour)
  return sentences, token_lists, y_labels 

# Experiment 1 repeated with data augmentation
def experiment_2():
  sentences = read(DRIVE_LOCATION + 'Data/augmented_stopwords/' + 'x_cwe_bert_aug.pickle')
  y_labels = read(DRIVE_LOCATION + 'Data/augmented_stopwords/' + 'y_cwe_bert_aug.pickle')
  token_lists = [word_tokenize(s) for s in sentences]
  return sentences, token_lists, y_labels 

# Clustering repeated on corrected labeling of samples by the clinician (no data augmentation)
def experiment_3():
  mri_report_corrected_df = pd.read_csv(DRIVE_LOCATION + 'Data/mri_report_corrected_df.csv',index_col=0)
  sentences, token_lists, y_labels = pre_process(mri_report_corrected_df.report_text, mri_report_corrected_df.tumour)
  return sentences, token_lists, y_labels 

# Clustering repeated on corrected labeling of samples by the clinician (with data augmentation)
def experiment_3_with_aug():
  sentences = read(DRIVE_LOCATION + 'Data/augmented_corrected_samples/' + 'x_cwe_bert_aug.pickle')
  sentences = sentences.drop([6333,7659,7660])
  sentences.reset_index(drop=True, inplace=True)
  y_labels = read(DRIVE_LOCATION + 'Data/augmented_corrected_samples/' + 'y_cwe_bert_aug.pickle')
  y_labels = y_labels.drop([6333,7659,7660])
  y_labels.reset_index(drop=True, inplace=True)
  token_lists = [word_tokenize(s) for s in sentences]
  return sentences, token_lists, y_labels 

if __name__ == '__main__':

    DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MSc Project/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=2)
    parser.add_argument('--method', default='TFIDF')
    parser.add_argument('--experiment', default='experiment_1') # {experiment_1, experiment_2, experiment_3, experiment_3_with_aug, experiment_4}
    parser.add_argument('--cluster_method', default='kmeans') # {kmeans, birch, GMM}
    args = parser.parse_args()

    sentences = None
    token_lists = None 
    y_labels = None
    name = None

    if(str(args.experiment) == 'experiment_1'):
      sentences, token_lists, y_labels = experiment_1()
    elif(str(args.experiment) == 'experiment_2'):
      sentences, token_lists, y_labels = experiment_2()
      name = 'experiment_1'
    elif(str(args.experiment) == 'experiment_3'):
      sentences, token_lists, y_labels = experiment_3()
    elif(str(args.experiment) == 'experiment_3_with_aug'):
      sentences, token_lists, y_labels = experiment_3_with_aug()
      name = 'experiment_1_corrected'
    else:
      sentences, token_lists, y_labels = experiment_3_with_aug()
      name = 'experiment_2'

    # Define the model object
    model = USLModel(sentences, token_lists, y_labels, args.method)
    model.cluster(args.cluster_method, args.method, args.k)
    
    # Save the model
    with open(DRIVE_LOCATION + "/Models/{}.file".format(model.id), "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    print(model.id)

    # Evaluate using metrics
    em = EvaluationMetrics(model, False)
    print(em.get_evaluation_metrics(model))
    lbs = em.get_label_df_new(model)
    print_metrics(lbs['ground_truth'], lbs['clusters'])

    # Uncomment if you want to display the UMAP cluster plot and word cloud for the clusters
    # Results will be saved to files/Results/{experiment_1, experiment_1_corrected, experiment_2}

    #if(name is not None):
    #  visualize and save img
    #  em.visualize(model, name)
    #  for i in range(model.k):
    #    em.get_wordcloud(model, token_lists, i, name)



