import asyncio
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from typing import Optional

from schemas import CATEGORICALS, COLS, NON_CATEGORICALS, PROPER_COLS_ORDER
from settings import CLASSIFIER_FILE, DATASET_RELATIVE_LOCATION, EXT, \
    TRAINING_DATASET_NAME, TESTING_DATASET_NAME

# TODO: Write tests (Mock predictor?)
dir_path = os.path.abspath(os.getcwd())
classifier = dir_path + CLASSIFIER_FILE + EXT


# TODO: enhance prediction by exponentiating fields fnlwgt, capital-loss,
# capital-gain in preprocessing.
def classify(sample: pd.DataFrame) -> np.ndarray:
    """Preprocess and classifies rows in sample.

    Args:
        sample (pandas.DataFrame): the sample rows to classify.

    Returns:
        numpy.ndarray containing the output classes.

    """
    preprocessed, _ = preprocess(sample)
    return joblib.load(classifier).predict(preprocessed)


def train_model(training_dataset: Optional[pd.DataFrame] = None):
    """Train the model on a given training set.

    Args:
        training_set (Optional[pands.DataFrame]): Will default to
        stored training set if not provided.

    """
    # TODO: test this.
    clf = joblib.load(classifier)

    if training_dataset is None:
        training_dataset = pd.read_csv(
            dir_path + DATASET_RELATIVE_LOCATION + TRAINING_DATASET_NAME,
            names=COLS
        )

    X_train, y_train = preprocess(training_dataset)
    clf.fit(X_train, y_train)
    joblib.dump(clf, classifier)


def evaluate_model_test_set_perf(testing_dataset:
                                 Optional[pd.DataFrame] = None) -> dict:
    """Evaluate the performance of the model against a test set.

    Args:
        testing_dataset (Optional[pandas.DataFrame]): the dataset to test
        against. Will default to stored test set if not provided.
        This should be kept separate from any training data.
    """
    # TODO: test this.
    clf = joblib.load(classifier)

    if testing_dataset is None:
        testing_dataset = pd.read_csv(
            dir_path + DATASET_RELATIVE_LOCATION + TESTING_DATASET_NAME,
            names=COLS
        )

    X_test, y_test = preprocess(testing_dataset)
    y_pred = clf.predict(X_test, y_test)
    comp = pd.concat([y_test, pd.DataFrame(y_pred)], axis=1)
    comp.columns = ['t', 'p']

    # To measure performance of the model, we use accuracy, precision
    # and recall.
    # It allows us to get an idea of the overfitting level of our model
    # to tune hyperparameters accordingly.
    # crossval score on the training set is not used here because the model
    # is overfitting when used on a to small subset of the data, meaning
    # that performance decreases when the best result on the test dataset
    # is not yet reached.
    # We used validation/train split of 0.3.
    # Maybe with a smaller validation proportion it would be
    # a better indicator.
    unaccurate = comp['t'] != comp['p']
    unaccurate_preds = np.where(unaccurate, comp['t'], np.nan)
    unaccurate_preds = unaccurate_preds[~np.isnan(unaccurate_preds)]
    accuracy = 1 - len(unaccurate_preds) / len(y_test)
    true_pos = ((comp['t'] + comp['p']) == 2).sum()
    false_pos = (comp['p'] == 1).sum() - true_pos
    false_negs = (comp['t'] == 1).sum() - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_negs)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def preprocess(dataset: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataset to make it compliant with predictor input.

    Args:
        dataset (pandas.DataFrame): the 'raw' dataset to format.

    Returns:
        pandas.DataFrame.

    """

    # We need to preprocess the data in order to input the classifier with
    # meaningful data: We represent categorical variables as dummies.
    # However this representation has a drawback: we will not be able to
    # handle values for categorical data that are not present in the
    # training set (example: new native country)).
    dataset.sample(frac=1)
    y_train, X_train = (dataset.get('class'), dataset.drop('class', 1)
                        if 'class' in dataset.columns else dataset)

    if y_train:
        y_train = (y_train.replace(' <=50K', 0).replace(' >50K', 1)
                   .replace(' <=50K.', 0).replace(' >50K.', 1))
    categorical_data = X_train[CATEGORICALS]
    categorical_as_dummies = pd.get_dummies(categorical_data,
                                            prefix=CATEGORICALS)
    # TODO: make this more easily readable.
    cat_col = [(name, [0]*len(dataset)) for name in PROPER_COLS_ORDER
               if name not in categorical_as_dummies.columns
               and name not in NON_CATEGORICALS]

    for name, col in cat_col:
        df = pd.DataFrame({name: col})
        categorical_as_dummies = pd.concat([categorical_as_dummies, df],
                                           axis=1)

    non_categorical_data = X_train[NON_CATEGORICALS]
    X_train = pd.concat([non_categorical_data, categorical_as_dummies], axis=1)
    X_train = X_train[PROPER_COLS_ORDER]
    return X_train, y_train
