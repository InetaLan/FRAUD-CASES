# FRAUD-CASES

## 1. Overview

The aim of study is to identify the most commonly used machine learning methods for detecting insurance claims fraud and apply them to the given data. Main objectives: conduct a literature analysis, prepare a dataset for the application of machine learning methods, implement chosen techniques, evaluate the classifiers performance and visualize the results.

## 2. Data
The data used in the study consist of motor insurance claims registered between 1 January 2023 and 31 December 2023 of physical insurers accidents in Lithuania. 
Dataset contains variouse attributes about claim case (status, type, description, reserved amount), claim event (type, date, city), insured object (vehicle model, make) and insurer.

In order to detect potential fraud cases using machine learning classifiers, a new fraud flag attribute was created to identify whether cases was determened as fraudulent or not in history by experts team.

The initial dataset consists of 17 556 rows and 24 columns. There is a strong class imbalance (only 4% of the cases are fraud cases), which may have an impact on the poor model classification.

### 2.1. Data preparation
Strong outliers were removed using IQR (Interquartile range) method (coefficient 3.5) seperatly for each class (fraud or non-fraud). 

Some categorical attributes, e.g. Vehicle model, could obtain a lot of unique values, thus possibly causing "_curse of dimensionality_" after encoding. A series of experiments showed that dimentionality reduction (UMAP, PCA, t-SNE) does not perform well on such dataset. For that reason feature engineering was used: only 5 most frequent values in non-fraudulent cases and values repeaded more than 3 times in fraudulent cases were retained, and the rest of values have been renamed as "_OTHER_". Latter on _OneHotEncoder_ was applied.

### 2.2. Event description
It was discovered that using event description could imporve model performance, but this feature requered NLP (Natural Language Preprocessing) including: text tokenization, stop words (lithuanian) and symbols removing, lemmatization and standartization. There was issue with Vehile brand name standartization - resolved by leaving words from brands list in the original form. The importance of each token was evaluated by _TF-IDF_ (term frequency - inverse document frequency) score. 
