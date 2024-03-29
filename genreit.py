#!/usr/bin/env python
import click
import json
import pickle
import os

import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.metrics import f1_score
from langdetect import detect

from model import Model
from preprocessing import preprocess_data, regularize_text


def prepare_model(model, estimator, data_file, subset, text_vars, genres_var, recombine_vertically, min_genre_count,
                  sep, max_tfidf, estimator_params, save=True, split=0.0, validate=True, random_state=412, **kwargs):
    """
    Preprocesses data read from data_file, extracts TF-IDF features, fits 'estimator' with them and finally
    calculates the threshold that maximizes the f1 score in the training/validation data.

    :param model: Model object.
    :param estimator: Object with fit and predict_proba methods.
    :param data_file: Str.
    :param subset: List of variables to extract from data_file.
    :param text_vars: List of text vars in subset.
    :param genres_var: Str. Name of genres var.
    :param recombine_vertically: dict. Maps variables that may be used as alternative descriptors.
    :param min_genre_count: float. Represents the number or proportion of total samples that a genre needs to have
    as its frequency to be kept in the "target_genres" variable.
    :param sep: str. Separator of data_file.
    :param max_tfidf: int or list. Max TF-IDF features for each text_var.
    :param estimator_params: dict. Parameters to pass to the estimator.
    :param save: bool. Whether to save trained model.
    :param split: float. Proportion of the data to be used as test. Use only if benchmarking.
    :param validate: bool. Whether to also split a validation set.
    :param random_state: int.
    :return: model object and dict with all data subsets.
    """
    print(f"Preprocessing {data_file}...")
    data, binarizer, vectorizer = preprocess_data(file=data_file, subset=subset, text_vars=text_vars,
                                                  genres_var=genres_var, recombine_vertically=recombine_vertically,
                                                  min_genre_count=min_genre_count, sep=sep, split=split,
                                                  validate=validate, max_tfidf=max_tfidf, random_state=random_state)
    model.binarizer = binarizer
    model.vectorizer = vectorizer
    print("Done.")

    print(f"Training {estimator}...")
    model.fit(data['X_train'], data['y_train'], estimator, **estimator_params)
    if split <= 0:
        model.set_best_threshold(data['X_train'], data['y_train'])
    else:
        model.set_best_threshold(data['X_test'], data['y_test'])
    print("Done.")

    if save:
        if not os.path.isdir('models'):
            os.mkdir('models')

        save_path = "models/" + model.estimator_name + ".pickle"
        pickle_out = open(save_path, 'wb')
        pickle.dump(model, pickle_out)
        pickle_out.close()
        print(f"Model saved to {save_path}")

    return model, data


def preprocess_input(title, overview, vectorizer):
    """
    Extract TF-IDF features from a single title and overview input.
    :param title: Str. Movie title.
    :param overview: Str. Movie plot.
    :param vectorizer: Fitted sklearn.feature_extraction.text.TfidfVectorizer object.
    :return: Tf-IDF features matrix.
    """
    # Clean text
    stop_words = stopwords.words("english")
    title = regularize_text(title, stop_words)
    overview = regularize_text(overview, stop_words)

    # Get TF-IDF features
    title = vectorizer['title'].transform([title])
    overview = vectorizer['overview'].transform([overview])

    # Merge features horizontally
    features = hstack((title, overview))

    return features


def predict_genres(X, model):
    """
    Predicts and returns the labels for a set of predictors X.
    :param X: Matrix.
    :param model: Model object.
    :return: Matrix.
    """
    preds = model.predict(X)
    return model.binarizer.inverse_transform(preds)[0]


def benchmark_model(config):
    """
    Processes data, fits model and prints both its f1 score and the f1 score of an equivalent random model.
    Used to test how well a certain model is performing.
    :param config: dict. Extracted from config file.
    """
    # Initial set-up
    config = parse_config(config)
    print(f'Benchmarking {config["estimator"]} on data {config["data_file"]}...')
    model = Model()

    # Process data, fit model and get its score
    model, data = prepare_model(model=model, save=False, split=0.2, validate=True, **config)
    model_score = model.score(data['y_val'], data['X_val'])

    # Get score of a random model
    random_result = (np.random.rand(data['y_val'].shape[0], data['y_val'].shape[1]) >= model.threshold).astype(int)
    random_score = f1_score(data['y_val'], random_result, average='micro')

    # Print results
    print('Model "{}" f1 score: {:.3f}'.format(model.estimator_name, model_score))
    print('A random genre assignation will have resulted in an f1 score of {:.3f}'.format(random_score))
    print('This means a difference of {:.3f}'.format(model_score-random_score))


def parse_config(config):
    """
    Loads config file.
    :param config: Str or Path to json file.
    :return: dict.
    """
    try:
        with open(config) as json_file:
            params = json.load(json_file)
    except json.JSONDecodeError:
        print("Unable to load config file")
        raise
    return params


def classify_movie(title, plot, model_file=None, config=None, save=True, force=False, model=None, verbose=False):
    """
    Given a title and plot input, classify it. If there's no model available to do so, train one.
    :param title: Str. Movie title.
    :param plot:  Str. Movie plot.
    :param model_file: Str or Path. Location of the .pickle model file.
    :param config: Str or Path. Location of the corresponding config file. Not used if a model manages to be loaded.
    :param save: bool. Whether to save the trained model.
    :param force: bool. If true, no model will be try to be loaded, it will use the config file to train one directly.
    The new model might overwrite an old one.
    :param model: loaded model.
    :return: list. Predictions for genres of the movie.
    """
    # Try to load ready-made model
    if model is None:
        model = Model(estimator_file=model_file)
        error = False
        if not force:
            try:
                model.load()
            except FileNotFoundError:
                error = True

        # If model is not found or the force flag is activated, train from data
        if error or force:
            config = parse_config(config)
            print(f'Model "{model_file}" not found. Training from {config["data_file"]}...')
            model, _ = prepare_model(model=model, save=save, **config)

    # Preprocess input data
    X = preprocess_input(title, plot, model.vectorizer)

    # Make predictions
    preds = predict_genres(X, model)

    # Print results
    if verbose:
        if len(preds) > 0:
            print("Predicted genres:", preds)
        else:
            print(f'Error: Movie "{title}" could not be classified with any genre. Try being more explicit in its plot.')

    return preds


@click.command()
@click.option('-title', '-t', default=None, help='Title of movie to classify')
@click.option('-description', '-d', default=None, help='Plot of movie to classify')
@click.option('-model_file', default='models/LogisticRegressionCV.pickle',
              help='Path to model Pickle file.')
@click.option('-config', default='configs/default.json',
              help='Path to config file if a model needs to be trained.')
@click.option('--not-save', is_flag=True, default=True, help='Do not save the model')
@click.option('--benchmark', is_flag=True, help='Test model performance or classify a movie')
@click.option('--force-train', is_flag=True, help='Train a model even if it is already available as a pickle file.')
def movie_classifier(title, description, model_file, config, not_save, benchmark, force_train):
    if benchmark:
        benchmark_model(config)
    else:
        # Input checking
        if title is None or description is None:
            raise ValueError("Movie title and plot must be provided in order to classify it.")
        elif len(description) < 10:
            raise ValueError("Please specify at least 10 characters for plot.")
        elif detect(description) != "en":
            raise ValueError("Movie title and plot must be in english")

        classify_movie(title, description, model_file, config, not_save, force_train)


if __name__ == '__main__':
    movie_classifier()
