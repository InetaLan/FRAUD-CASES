# Fraudulent claim cases

## 1. Overview
The aim is to apply Random Forest Classifiaction for detecting fraudulent insurance claims fraud. Main objectives: conduct a literature analysis, prepare a dataset for the application of machine learning methods, implement chosen techniques and evaluate the classifiers performance.

## 2. Data
The data used in the study consist of motor insurance claims registered between 1 January 2023 and 31 December 2023 of physical insurers accidents in Lithuania. 
Dataset contains variouse attributes about claim case (status, type, description, reserved amount), claim event (type, date, city), insured object (vehicle model, make) and insurer.

In order to detect potential fraud cases using machine learning classifiers, a new fraud flag attribute was created to identify whether cases was determened as fraudulent or not in history by experts team.

The initial dataset consists of 17 556 rows and 24 columns. There is a strong class imbalance (only 4% of the cases are fraud cases), which may have an impact on the poor model classification.

> [!NOTE]
> Data preparation code is in make_data file.

#### 2.1. Feature engineering
Strong outliers were removed using IQR (Interquartile range) method (coefficient 3.5) seperatly for each class (fraud or non-fraud). 
Some categorical attributes, e.g. Vehicle model, could obtain a lot of unique values, thus possibly causing "_curse of dimensionality_" after encoding. A series of experiments showed that dimentionality reduction (UMAP, PCA, t-SNE) does not perform well on such dataset. For that reason feature engineering was used: only 5 most frequent values in non-fraudulent cases and values repeaded more than 3 times in fraudulent cases were retained, and the rest of values have been renamed as "_OTHER_". Latter on _OneHotEncoder_ was applied.

#### 2.2. Event description
It was discovered that using event description could imporve model performance, but this feature requered NLP (Natural Language Preprocessing) including: text tokenization, stop words (lithuanian) and symbols removing, lemmatization and standartization. There was issue with Vehile brand name standartization - resolved by leaving words from brands list in the original form. The importance of each token was evaluated by _TF-IDF_ (term frequency - inverse document frequency) score. 

#### 2.3. Class balansing
Classifier was trained (and then evaluated) on 5 different training sets: original, balanased with SMOTE, RUS, ENN and SMOTE-ENN. The best results were reached with ENN balansing technique. 

## 3. Model implementation
- Atfer all preparation steps, final dataset was splitted into Training and Testing sets (20 % for testing, stratified by class attribute).
- Training dataset was balansed using ENN technique.
- Optimal hyperparameters were selected with _GridSearchCV_ on 30% of Training data (due to long computation).
```
def train_random_forest(Train_X, Train_Y):
    param_grid = {
        "n_estimators": [100, 200, 500, 1000],
        "max_depth":  [None, 10, 30, 50, 70, 100],
        "min_samples_split": np.arange(2, 5, 1),
        "max_features": ["sqrt", "log2"]
        }
    forest = RandomForestClassifier()
    scoring = ('f1_macro', 'recall_macro')

    grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, n_jobs=-1, scoring=scoring, refit='recall_macro', return_train_score=True)
    grid_search.fit(Train_X, Train_Y)

    best_params = grid_search.best_params_

    best_RF = grid_search.best_estimator_
    return best_RF

best_rf = train_random_forest(Val_X, Val_Y)
```
- Optimal classifier then fitted to _full_ Training set, predictions evaluated on Training and Testing sets. Feature importance was looked at out of curiosity, but not taken into account anywhere further.
```

def evaluate_classifier(classifier, Train_X, Train_Y, Test_X, Test_Y):
    classifier.fit(Train_X, Train_Y)

    feature_importances = classifier.feature_importances_
    feature_names = Train_X.columns
    importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(20)
    print("Feature importances (sorted):")
    for index, row in importances_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    y_pred_train = classifier.predict(Train_X)
    y_pred_test = classifier.predict(Test_X)
    print("Classification Report - Training Set:\n", classification_report(Train_Y, y_pred_train))
    print("Classification Report - Testing Set:\n", classification_report(Test_Y, y_pred_test))

    cm_train = confusion_matrix(Train_Y, y_pred_train)
    cm_test = confusion_matrix(Test_Y, y_pred_test)

    # plot confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(25, 10))
    
    cm_display_train = ConfusionMatrixDisplay(cm_train)
    cm_display_train.plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title('Training set', fontsize=24)
    ax[0].set_xlabel('Prediction', fontsize=20)
    ax[0].set_ylabel('True class', fontsize=20)
    ax[0].tick_params(labelsize=16)

    cm_display_test = ConfusionMatrixDisplay(cm_test)
    cm_display_test.plot(ax=ax[1], cmap='Blues', values_format='d')
    ax[1].set_title('Testing set', fontsize=24)
    ax[1].set_xlabel('Prediction', fontsize=20)
    ax[1].set_ylabel('True class', fontsize=20)
    ax[1].tick_params(labelsize=16)

    plt.show()

evaluate_classifier(
    best_rf,
    Train_X, Train_Y,
    Test_X, Test_Y,
    )
```
## 4. Results

| Class | Precision | Recall | F1 | N |
|:-----|:--------:|:------:|:------:|:------:|
| Non-fraud | 0.97| 0.98 | 0.98 | 2 265 |
| Fraud | 0.74| 0.62|0.68|176|
|  | ||||
|Accuracy | ||0.96|2 441|
|  | ||||
|Macro Avg |0.86 |0.80|0.83|2 441|
|Weighted Avg|0.95|0.96|0.96|2441|
