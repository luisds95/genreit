import pickle
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


class Model:
    def __init__(self, estimator_file=None):
        self.estimator = None
        self.estimator_file = estimator_file
        self.estimator_name = None
        self.threshold = 0.5
        self.binarizer = None
        self.vectorizer = None

    def load(self):
        if self.estimator_file is None:
            raise ValueError('Specify an estimator_path for loading a pre-trained model.')

        # Load file
        file = open(self.estimator_file, "rb")
        model = pickle.load(file)

        # Set variables from loaded model
        self.estimator = model.estimator
        self.estimator_name = model.estimator_name
        self.binarizer = model.binarizer
        self.vectorizer = model.vectorizer

        file.close()

    def fit(self, X, y, estimator='logistic', n_jobs=-1, **estimator_params):
        # Select estimator
        if estimator == 'logistic':
            estimator = LogisticRegressionCV(**estimator_params)
        elif callable(estimator):
            estimator = estimator(**estimator_params)
        else:
            raise NotImplementedError(f'Estimator "{estimator}" not yet implemented!')

        # Build into OneVsRestClassifier and fit
        self.estimator = OneVsRestClassifier(estimator=estimator, n_jobs=n_jobs)
        self.estimator.fit(X, y)
        self.estimator_name = type(estimator).__name__

    def predict(self, X=None, probas=None, t=None):
        # Parameter setting and error checking
        if X is None and probas is None:
            raise TypeError("Either X or probas must be provided.")
        if t is None:
            t = self.threshold

        # Get probabilities matrix
        if probas is None:
            probas = self.estimator.predict_proba(X)

        # Set to 1 of probability is >= threshold
        return (probas >= t).astype(int)

    def score(self, y, X=None, probas=None, t=None):
        preds = self.predict(X, probas, t)
        return f1_score(y, preds, average='micro')

    def set_best_threshold(self, X, y, precision=0.01, max_t=0.5, min_t=None, bias=0):
        # Parameter setting and error checking
        if min_t is None:
            min_t = precision
        if min_t > max_t:
            raise ValueError("Minimum threshold needs to be less than maximum.")

        # Get probas and score for current threshold
        probas = self.estimator.predict_proba(X)
        best_score = self.score(y, probas=probas)

        # Loop to try to find a better threshold
        for t in np.arange(min_t, max_t, precision):
            score = self.score(y, probas=probas, t=t)
            if score >= best_score:
                best_score = score
                self.threshold = t + bias


