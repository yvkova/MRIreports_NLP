# MRIreports_NLP
This study tests several embedding and clustering methods for classifying free-text reports on magnetic resonance imaging (MRI) scans of the brain of patients examined for vestibular schwannomas in unsupervised manner. With two clusters, the purpose was to see the performance of the tested methods when classifying the reports into those mentioning a tumour (positive class) and not (negative class). With three clusters, the purpose was to see if the approach can allow discovering incidental findings and errors.

The methodology and results of the study are summarised in the [MRIreports_NLP.pdf](../main/MRIreports_NLP.pdf)


## Getting started

- The project was developed on Colab with DRIVE_LOCATION = '/content/gdrive/MyDrive/Colab Notebooks/MRI Project/'
- The dataset is not included (awaits ethical approval). The code can be tested on a csv file containing report texts and their labels. 
- Requires downloading BioSentVec_PubMed_MIMICIII-bigram_d700.bin, bioword2vec_model.file and BioWordVec_PubMed_MIMICIII_d200.vec.bin and saving then to the /files directory. These files are used by vec_embeddings.py.
- Links: <https://github.com/ncbi-nlp/BioWordVec>, <https://github.com/ncbi-nlp/BioSentVec>

## Getting familiar with the files

Primary classes:

- main.py: the main script launching the experiments; change the parameters based on the scenario required to run.
- preprocess.py: performs preprocessing and cleaning of the original dataset.
- vec_embeddings.py: contains all the vector embeddings used in the experiments.
- usl_model.py: contains the model creation code, i.e. creates and runs clustering models.
- evaluation_metrics.py: calculates the metrics scores from the model performance.

Helper classes:

- data_augmentation.py: performs data augmentation on the original dataset.
- pca.py: utility class for dimensionality reduction (PCA).
- utils.py: helper functions such as read().
- autoencoder.py: utility class for autoencoder logic.
- eda.py: contains the code essential for the exploratory data analysis (EDA) conducted in the study as outlined in the report.
