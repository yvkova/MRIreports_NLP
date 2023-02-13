# for data augmentation
import pandas as pd
import pickle
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MSc Project/'

class TextAugmentationNlPaug:

  x_train = None
  y_train = None
  textAugmentNlpaugConfig = None
  path_balanced ='Data/augmented/'
  


  def __init__(self, x_train, y_train, path):
    self.x_train = x_train
    self.y_train = y_train
    self.textNlpaugConfig = TextAugmentNlPaugConfig()
    self.path_balanced = path


  def augmentContextWordEmb(self):
    aug_sentences=[]
    aug_sentences_labels=[]
    aug_sentences_index = []
    for i in self.x_train.index:
      if self.y_train[i]==1:
        temps=self.textNlpaugConfig.context_word_emb_aug.augment(self.x_train[i])[0]
        aug_sentences.append(temps)
        aug_sentences_labels.append(1)
        aug_sentences_index.append(i)
    self.saveToFile(aug_sentences, aug_sentences_labels, aug_sentences_index, 'cwe_bert')
    return (aug_sentences, aug_sentences_labels, aug_sentences_index, self.x_train, self.y_train)

  
  def augmentBackTranslation(self):
    aug_sentences=[]
    aug_sentences_labels=[]
    aug_sentences_index = []
    for i in self.x_train.index:
      if self.y_train[i]==1:
        temps=self.textNlpaugConfig.back_translation_aug.augment(self.x_train[i])[0]
        aug_sentences.append(temps)
        aug_sentences_labels.append(1)
        aug_sentences_index.append(i)
    self.saveToFile(aug_sentences, aug_sentences_labels, aug_sentences_index, 'bt')
    return (aug_sentences, aug_sentences_labels, aug_sentences_index, self.x_train, self.y_train)

  def saveToFile(self,aug_sentences,aug_sentences_labels,aug_sentences_index, prefix_name):
    self.writeToFile(aug_sentences, self.path_balanced, prefix_name +'_aug_sentences')
    self.writeToFile(aug_sentences_labels, self.path_balanced, prefix_name +'_aug_sentence_labels')
    self.writeToFile(aug_sentences_index, self.path_balanced, prefix_name +'_aug_sentence_index')

    self.x_train= self.x_train.append(pd.Series(aug_sentences),ignore_index=True)
    self.y_train= self.y_train.append(pd.Series(aug_sentences_labels),ignore_index=True)

    self.writeToFile(self.x_train, self.path_balanced, 'x_' + prefix_name + '_aug')
    self.writeToFile(self.y_train, self.path_balanced, 'y_' + prefix_name + '_aug')
  
  def writeToFile(self, object_to_write, path, name):
    location = DRIVE_LOCATION + path + name
    with open(location +'.pickle', 'wb') as output:
      pickle.dump(object_to_write, output)

  def augmentSynonym(self):
    config = self.textNlpaugConfig
    aug_sentences=[]
    aug_sentences_labels=[]
    aug_sentences_index = []
    for i in self.x_train.index:
      if self.y_train[i]==1:
        temps= config.synonym_aug.augment(self.x_train[i],n=config.synonym_aug__n)
        for sent in temps:
          aug_sentences.append(sent)
          aug_sentences_labels.append(1)
          aug_sentences_index.append(i)
    self.saveToFile(aug_sentences, aug_sentences_labels, aug_sentences_index, 'syn')
    return (aug_sentences, aug_sentences_labels, aug_sentences_index, self.x_train, self.y_train)

  def convertAugmentedSentences(self, augmented_sentences, augmented_labels):
    self.x_train= self.x_train.append(pd.Series(augmented_sentences),ignore_index=True)
    self.y_train= self.y_train.append(pd.Series(augmented_labels),ignore_index=True)
    return (self.x_train, self.y_train)
    
    
class TextAugmentNlPaugConfig:

  synonym_aug = None
  synonym_aug__n = None
  back_translation_aug = None
  context_word_emb_aug = None

  def __init__(self):
    self.synonym_aug = naw.SynonymAug(aug_src='wordnet',aug_max=3)
    self.synonym_aug__n = 2 #4

    self.back_translation_aug = naw.BackTranslationAug(
                                  from_model_name='facebook/wmt19-en-de', 
                                  to_model_name='facebook/wmt19-de-en'
                              )
    self.context_word_emb_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")

def augment(path, x, y):
  textAugNlPaug = TextAugmentationNlPaug(path, x, y)
  print(" Starting 1st augmentation: back translation...")
  (aug_sentences, aug_sentences_labels, aug_sentences_index, x, y) = textAugNlPaug.augmentBackTranslation()
  print(" 1st augmentation: back translation Done!")

  print(" Starting 2nd augmentation: synonym augmentation...")
  (aug_sentences, aug_sentences_labels, aug_sentences_index, x, y) = textAugNlPaug.augmentSynonym()
  print(" 2nd augmentation: synonym augmentation Done!")

  print(" Starting 3rd augmentation: contexual word embedding(bert based)...")
  (aug_sentences, aug_sentences_labels, aug_sentences_index, x, y) = (aug_sentences, aug_sentences_labels, aug_sentences_index, x, y) = textAugNlPaug.augmentContextWordEmb()
  print(" 3rd augmentation: contexual word embedding(bert based) Done!")
  return x, y

