# MRIreports_NLP
This study tests several embedding and clustering methods for classifying free-text reports on magnetic resonance imaging (MRI) scans of the brain of patients examined for vestibular schwannomas in unsupervised manner. With two clusters, the purpose was to see the persformance of the tested methods when classying the reports into those mentioning a tumour (positive class) and not (negative class). With three clusters, the purpose was to see if the approach can allow discovering incidental findings and errors.

The discription and results of the study are summarised in the [MRIreports_NLP.pdf](../main/MRIreports_NLP.pdf)


## Getting started

- The project was developed on Colab.
- The dataset is not included (awaits ethical aproval). The project can be testes on a scv file containing ID and text of documents. 
- Requires downloading BioSentVec_PubMed_MIMICIII-bigram_d700.bin, bioword2vec_model.file and BioWordVec_PubMed_MIMICIII_d200.vec.bin and saving then to the /files directory. These files are used by vec_embeddings.py.
- Links: <https://github.com/ncbi-nlp/BioWordVec>, <https://github.com/ncbi-nlp/BioSentVec>

## Pre-requisites

- On Google Colab, follow the instructions of PRE-REQUISITES.ipynb.
- Open terminal and execute the main.py file.

```
python main.py
```

## Getting familiar with the files

Primary classes:

- main.py : the main script launching the experiments; change the parameters based on the scenario required to run.
- preprocess.py: performs preprocessing and cleaning of the original dataset.
- usl_model.py: contains the model creation code, i.e. creates and runs clustering models.
- vec_embeddings.py: contains all the vector embeddings used in the experiments.
- evaluation_metrics.py: calculates the metrics scores from the model performance.

Helper classes:

- data_augmentation.py: contains the code to perform data augmentation on the original dataset.
- pca.py: utility class for dimensionality reduction (pca).
- utils.py: helper functions such as read().
- autoencoder.py: utility class for autoencoder logic.
- eda.py: contains the code essential for the EDA conducted in the study.
