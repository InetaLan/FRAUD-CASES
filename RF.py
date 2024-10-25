import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,ConfusionMatrixDisplay)

from make_data import Val_X, Val_Y, Train_X, Train_Y, Test_X, Test_Y


def train_random_forest(Train_X, Train_Y):
    param_grid = {
        "n_estimators": [100, 200, 500, 1000]
        "max_depth":  [None, 10, 30, 50, 70, 100],
        "min_samples_split": np.arange(2, 5, 1),
        "max_features": ["sqrt", "log2"]
    }
    forest = RandomForestClassifier(class_weight="balanced_subsample")
    scoring = ('f1_macro', 'recall_macro')

    grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, n_jobs=-1, scoring=scoring, refit='recall_macro', return_train_score=True)
    grid_search.fit(Train_X, Train_Y)

    best_params = grid_search.best_params_
    best_RF = grid_search.best_estimator_

    return best_RF

best_rf = train_random_forest(Val_X, Val_Y)

def evaluate_classifier(classifier, Train_X, Train_Y, Test_X, Test_Y):
    classifier.fit(Train_X, Train_Y)

    feature_importances = classifier.feature_importances_
    feature_names = Train_X.columns
    importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(20)
    print("Top 15 Feature importances (sorted):")
    for index, row in importances_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    y_pred_train = classifier.predict(Train_X)
    y_pred_test = classifier.predict(Test_X)
    print("Classification Report - Training Set:\n", classification_report(Train_Y, y_pred_train))
    print("Classification Report - Testing Set:\n", classification_report(Test_Y, y_pred_test))

    calculate confusion matrices for train and test data
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
    Test_X, Test_Y)
