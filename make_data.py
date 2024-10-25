import pandas as pd
import numpy as np
import simplemma

from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import model_selection
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

na_values_list = ["-", " ", "(n/z)"]
DATA_2023 = pd.read_excel("3cf4387e-a552-4d31-987c-07927aff55a8.xlsx", na_values=na_values_list)

# Restricting only needed data
data = DATA_2023.loc[
    (DATA_2023['INSURER.PERSON_TYPE_NAME'] == "P") & 
    (DATA_2023['CLM_EVENT.EVENT_COUNTRY'] == "LTU")  & 
    (DATA_2023['CLM_EVENT.D_EVENT'] >= pd.to_datetime('2023-01-01')) &
    (DATA_2023['AGREEMENT.PERIOD_DAYS'] != 1096) &
    (DATA_2023['Claims_paid'] < 60000)]

columns_to_drop = ['CLM_CASE.#KEY', 'CLM_EVENT.#KEY', 'OBJECT.#KEY', 'OBJECT.OBJECT_NAME', 'AGREEMENT.#KEY','INSURER.#KEY', 'EMP.#KEY',
                   'OBJECT.OBJECT_TYPE_NAME', 'CLM_CASE.NOTIFICIATION_TYPE_NAME', 'CLM_EVENT.EVENT_COUNTRY_NAME', 
                   'CL_HANDLE_SPEED.BASKET_NAME','AGREEMENT_ATTRIBUTES.TG_PERIOD_CODE', 
                   'CLM_EVENT.EVENT_COUNTRY', 'CLM_EVENT.DOMESTIC_OR_ABROAD', 'INSURER.PERSON_TYPE_NAME', 
                   'AGREEMENT.ISSUE_YEAR_AND_MONTH', 'AGREEMENT.START_YEAR_AND_MONTH', 'AGREEMENT.END_WITH_CANCELLATION_YEAR_AND_MONTH']
data.drop(columns=columns_to_drop,  axis=1, inplace=True)
data = data.dropna()
print("Data shape", data.shape)

data['Days_from_event_till_agr_end'] = (data['AGREEMENT.D_TO_ACTUAL'] - data['CLM_EVENT.D_EVENT']).dt.days
data['Days_from_agr_start_till_event'] = (data['CLM_EVENT.D_EVENT'] - data['AGREEMENT.D_FROM']).dt.days
data['Days_from_start_till_end'] = (data['AGREEMENT.D_TO_ACTUAL'] - data['AGREEMENT.D_FROM']).dt.days

datetime_columns_drop = ['CLM_CASE.D_NOTIFICATION', 'CLM_EVENT.D_EVENT', 'CLM_EVENT.D_EVENT_REPORT', 'AGREEMENT.D_ISSUED', 
                         'AGREEMENT.D_FROM', 'AGREEMENT.D_TO_ACTUAL']
data.drop(columns=datetime_columns_drop, inplace=True)


""" Creating fraudulent cases flag using historical data """
excel_df = pd.read_excel('Book3.xlsx',sheet_name='Sheet1')
data = pd.merge(data, excel_df[['Claim no.']], how='left', left_on='CLM_CASE.CASE_NUM', right_on='Claim no.')
data['label'] = '0'
data.loc[data['Claim no.'].notnull(), 'label'] = '1'
data.drop('Claim no.', axis=1, inplace=True)
data.drop('CLM_CASE.CASE_NUM', axis=1, inplace=True)
print("Desctibution of classes in original data: ",Counter(data['label']))


""" Removing outliers seperatly for each class """
def drop_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.5 * IQR
    upper_bound = Q3 + 3.5 * IQR
    # Only keep rows in dataframe that do not have outliers.
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]

data_label_0 = data[data['label'] == '0']
data_label_1 = data[data['label'] == '1']

for column in columns_to_check:
    data_label_0 = drop_outliers(data_label_0, column)
for column in columns_to_check:
    data_label_1 = drop_outliers(data_label_1, column)
data = pd.concat([data_label_0, data_label_1])


""" Mofify large categorical data """
def modify_data_by_top_values(data, column, label_col='label'):
    grouped = data.groupby([column, label_col]).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
    top_n_values = grouped[grouped[label_col] == '0'].nlargest(5, 'Count')[column].tolist()
    y_values = grouped[(grouped[label_col] == '1') & (grouped['Count'] > 3)][column].tolist()
    unique_values = list(set(top_n_values + y_values))
    data.loc[~data[column].isin(unique_values), column] = 'OTHER'
    return data

data = modify_data_by_top_values(data, 'CLM_EVENT.CITY_NAME')
data = modify_data_by_top_values(data, 'Vehicle make')
data = modify_data_by_top_values(data, 'Vehicle model')


"""Event description"""
with open('stopwords-lt.txt', 'r', encoding='utf-8') as file:
    stopwords_lt  = file.read()

def preprocess_text(text, stopwords_lt, brand_names):
    if pd.isna(text):
        text = ""
    tokens = word_tokenize(text)
    brand_names_lower = [name.lower() for name in brand_names]
    processed_tokens = []
    
    for token in tokens:
        if token.isalpha() and token.lower() not in stopwords_lt:
            if token.lower() in brand_names_lower:
                processed_tokens.append(token)
            else:
                processed_tokens.append(simplemma.lemmatize(token.lower(), lang=("lt", "en")))
    return " ".join(processed_tokens)

data['Processed_Text'] = data['CLM_CASE.CASE_DESCRIPTION'].apply(lambda x: preprocess_text(x, stopwords_lt, brand_names))
data.drop('CLM_CASE.CASE_DESCRIPTION', axis=1, inplace=True)
data['Word_Count'] = data['Processed_Text'].apply(lambda x: len(x.split()))

to_drop = ['Claims_paid', 'Reserved_amt', 'AGREEMENT.PERIOD_DAYS']
data.drop(columns=to_drop, inplace=True)

data['Processed_Text'] = data['Processed_Text'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() != 'tp'))
data['Words'] = data['Processed_Text'].str.split()

vectorizer = TfidfVectorizer(max_features = 30)
X_transformed = vectorizer.fit_transform(data['Processed_Text'])
print(X_transformed.shape)
print(vectorizer.get_feature_names_out())

# Print data example
dense_head = X_transformed[:5].todense()
df_head = pd.DataFrame(dense_head, columns=vectorizer.get_feature_names_out())
print("head of vectorized description")
print(df_head)

""" Encoding variables """
y = data['label']
X = data.drop(columns=['label', 'Processed_Text', 'Words'])

le = LabelEncoder()
y = le.fit_transform(y)
print(pd.Series(y).value_counts())

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(X[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_columns, index=X.index)
X = X.drop(columns=categorical_columns)
X_encoded = pd.concat([X, encoded_df], axis=1)

# Combining prepared encoded data and vectorized text
X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=vectorizer.get_feature_names_out())
X_final = pd.concat([X_encoded.reset_index(drop=True), X_transformed_df.reset_index(drop=True)], axis=1)
print("Shape of the final feature set: ", X_final.shape)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_final, y, test_size=0.20,stratify=y, random_state=123)

# Undesanmpling training data (dealing with huge class dissbalanse problem)
undersample = EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=5, n_jobs=-1)
Train_X, Train_Y = undersample.fit_resample(Train_X, Train_Y)
print(Counter(Train_Y))

Val_X, dev_x, Val_Y, dev_y = model_selection.train_test_split(Train_X,Train_Y, train_size=0.30, stratify=Train_Y, random_state=123)

