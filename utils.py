# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import pickle


DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MSc Project/'

def get_model_results(model):
  output_with_augmentation_NOAUGMETRICS=None
  output_std=None
  if(model =='K-MEANS'):
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_km_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_std_preprocess'))
  elif(model =='BIRCH'):
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_birch_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_birch_std_preprocess'))
  else:
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_gmm_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1/{}.file".format('output_gmm_std_preprocess'))
  
  return output_std, output_with_augmentation_NOAUGMETRICS

def get_corrected_model_results(model):
  output_with_augmentation_NOAUGMETRICS=None
  output_std=None

  if(model =='K-MEANS'):
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_km_corrected_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_kmeans_corrected'))
  elif(model =='BIRCH'):
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_birch_corrected_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_birch_corrected'))
  else:
    output_with_augmentation_NOAUGMETRICS = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_gmm_corrected_with_augmentation'))
    output_std = read(DRIVE_LOCATION + "/files/results/experiment_1_corrected/{}.file".format('output_gmm_corrected'))
  
  return output_std, output_with_augmentation_NOAUGMETRICS

def plot_horizontal_hist_metrics(model, metric, label, corrected=False):
  path = '/files/results/experiment_1/plots/{}'
  if(corrected == True):
    output_std, output_with_aug_NOAUGMETRICS = get_corrected_model_results(model)
    path = '/files/results/experiment_1_corrected/plots/{}'
  else:
    output_std, output_with_aug_NOAUGMETRICS = get_model_results(model)
  output_with_aug_NOAUGMETRICS['type']='with data augmentation'
  output_std['type'] = 'standard'
  df = output_with_aug_NOAUGMETRICS.append(output_std)

  sns.set_theme(style="whitegrid")
  # Draw a nested barplot by species and sex
  g = sns.catplot(
      data=df, kind="bar",
      y="Model", x=metric, hue="type",
      ci="None", palette="pastel", alpha=.6, height=7.5
  )
  g.despine(left=True)
  g.set_axis_labels(label, "Vector Embeddings")
  g.legend.set_title("Metric Calculated by")
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle((model + " "+label+ " Results"))

  dr = DRIVE_LOCATION + path.format(model + '_'+ label)
  g.savefig(dr)

def read(name):
    with open(name, 'rb') as data:
      model = pickle.load(data)
    return model

def write(object_to_write, path, name):
    with open(DRIVE_LOCATION + path + name, 'wb') as output:
      pickle.dump(object_to_write, output)

def read_data():
    mri_report_df = pd.read_excel(DRIVE_LOCATION + "Data/MRIreports.xlsx")
    mri_report_df.rename(columns={'Report Text': 'report_text', 'Tumour': 'tumour'}, inplace=True)
    return mri_report_df

def read_data(path):
    mri_report_df = pd.read_excel(path)
    mri_report_df.rename(columns={'Report Text': 'report_text', 'Tumour': 'tumour'}, inplace=True)
    return mri_report_df