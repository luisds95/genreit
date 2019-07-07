from collections import Counter
import json
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def read_data(file, subset, sep=","):
    return pd.read_csv(file, sep=sep, usecols=subset)


def json_genres_to_list(j, genres):
    # Select a json file a extract the field "name" from it
    j = json.loads(j)
    return [i['name'] for i in j if i['name'] in genres]


def regularize_text(text, stop_words):
    # Remove special characters and punctuation.
    if text is np.nan:
        return text

    text = re.sub('[^a-zA-Z]+', ' ', text)

    # Replace with lower
    text = text.lower()

    # Lemmatize so different gramatical forms of a word are represented as the same word
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text.split(' ') if word not in stop_words]
    text = ' '.join(text)

    # Remove double spaces
    text = text.replace("  ", " ")

    return text


def clean_data(data, text_vars, genres_var, min_genre_count=0.02, recombine_vertically=None):
    # Make json right
    data[genres_var] = data[genres_var].apply(lambda x: x.replace("'", '"'))

    # Get the frequency of each genre
    counter = Counter()
    for obs in data[genres_var]:
        obs_genres = json.loads(obs)
        counter.update([obs_genre['name'] for obs_genre in obs_genres])

    # Filter genres per minimum frequency
    target_genres = [genre for genre, count in counter.items() if count >= min_genre_count]

    # Get target genres per observation
    data_genres = [json_genres_to_list(j, target_genres) for j in data[genres_var]]

    # Now, convert them to dummies
    multi_label_binarizer = MultiLabelBinarizer(classes=target_genres).fit(data_genres)
    genre_dummies = multi_label_binarizer.transform(data_genres)
    data = data.drop([genres_var], axis=1)

    # If there are variables that may be used as alternative text sources,
    # add them as new observations
    if recombine_vertically is not None:
        # Merge genre_dummies back to the original dataframe to facilitate duplication
        data = data.merge(pd.DataFrame(genre_dummies, columns=target_genres), left_index=True, right_index=True)

        # Move variable "alternative" to below "original"
        for alternative, original in recombine_vertically.items():
            new_data = data.drop([original], axis=1)
            new_data = new_data[~new_data[alternative].isna()]
            new_data = new_data.rename({alternative: original})
            data = pd.concat((data, new_data), sort=False, ignore_index=True)
            data = data.drop([alternative], axis=1)

        # Separate target (genres) from predictors (text)
        genre_dummies = data[target_genres]
        data = data.drop(target_genres, axis=1)

    # Remove any duplicates
    duplicates = data.duplicated()
    data = data[~duplicates]
    genre_dummies = genre_dummies[~duplicates]

    # Regularize all text vars
    stop_words = stopwords.words("english")
    for var in text_vars:
        data[var] = data[var].apply(lambda x: regularize_text(x, stop_words))

    # Remove observations with missing values in any required text variable
    missings = data[text_vars].isna().any(axis=1)
    data = data[~missings]
    genre_dummies = genre_dummies[~missings]

    return data, genre_dummies, multi_label_binarizer


def split_sets(X, y, test_size=0.2, validate=True, shuffle=True, random_state=412):
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        shuffle=shuffle)

    # Split into train/test/validate sets
    if validate:
        validate_size = test_size / (1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validate_size,
                                                          random_state=random_state, shuffle=shuffle)
        return {'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test}

    # Or simply return train/test sets
    else:
        return {'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test}


def tf_idf_features(Xs, text_vars, max_features=None, **kwargs):
    new_X = {}
    vectorizers = {}

    # One vectorizer per variable
    for var, maxf in zip(text_vars, max_features):
        # Build vectorizer
        vectorizer = TfidfVectorizer(max_features=maxf, **kwargs)

        # Fit vectorizer on train and transform on every other dataset
        vectorizer.fit(Xs[0][var])
        vectorizers[var] = vectorizer

        new_X[var] = []
        for X in Xs:
            new_X[var].append(vectorizer.transform(X[var]))

    # Now, join the features of each text_var into a single matrix
    feature_matrices = []
    for i in range(len(Xs)):
        single_dataframe_features = [new_X[var][i] for var in text_vars]
        feature_matrices.append(hstack(single_dataframe_features))

    return feature_matrices, vectorizers


def preprocess_data(file, subset, text_vars, genres_var, recombine_vertically=None,
                    min_genre_count=0.02, sep=",", split=0.2, validate=True, max_tfidf=None, random_state=412):
    data = read_data(file=file, subset=subset, sep=sep)

    # Set an absolute value for min_genre_count
    if 0 < min_genre_count <= 1:
        min_genre_count *= data.shape[0]

    # Make sure text_vars is a list
    if not isinstance(text_vars, list):
        text_vars = [text_vars]

    # Check that max_tfidf and text_vars have the same length
    if max_tfidf is None:
        max_tfidf = [None for _ in text_vars]
    elif isinstance(max_tfidf, int):
        max_tfidf = [max_tfidf for _ in text_vars]
    else:
        assert len(text_vars) == len(max_tfidf), 'Max_features must have the same length as text_vars'

    # Cleaning
    data, genre_dummies, multi_label_binarizer = clean_data(data, text_vars, genres_var,
                                                            min_genre_count, recombine_vertically)

    # Split train/validation/test
    if split > 0:
        data = split_sets(data, genre_dummies, test_size=split, validate=validate,
                          shuffle=True, random_state=random_state)
    else:
        data = {'X_train': data, 'y_train': genre_dummies}

    # Compute tf-idf features
    Xs_keys = [k for k in data.keys() if k[0] == 'X']
    Xs = [data[k] for k in Xs_keys]
    tfidfs, vectorizers = tf_idf_features(Xs, text_vars, max_features=max_tfidf, max_df=0.8)

    # Replace them in the data dict
    for i, k in enumerate(Xs_keys):
        data[k] = tfidfs[i]

    return data, multi_label_binarizer, vectorizers
