"""Class that will make the data pipeline for the churn case study."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def grid_search(model, grid, X, y, cv=5, scaling=True):
    """Perform a grid search over the grid for the training data."""
    g = GridSearchCV(model, grid, n_jobs=1, cv=10)
    pl = ChurnPipeline(g, scaling)
    pl.fit(X, y)
    if scaling:
        print("Best parameters:", pl.Pipeline.steps[1][1].best_params_)
        return pl.Pipeline.steps[1][1].best_estimator_
    else:
        print("Best parameters:", pl.Pipeline.steps[0][1].best_params_)
        return pl.Pipeline.steps[0][1].best_estimator_


class ChurnPipeline():
    """Class that will make the data pipeline for the churn case study."""

    def __init__(self, model, scaling=True):
        """Instantiate the class."""

        self.Pipeline = self._init_pipeline(model, scaling)
        self.scaling = scaling

    def _init_pipeline(self, model, Scaling):
        """Create the pipeline object."""
        if Scaling:
            scaling = StandardScaler()
            return Pipeline([('scaler', scaling), ('model', model)])
        else:
            return Pipeline([('model', model)])

    def fit(self, X, y):
        """Fit the pipeline model to the training data."""
        self.Pipeline.fit(X, y)

    def predict(self, X):
        """Make predictions based on the X matrix."""
        return self.Pipeline.predict(X)

    def cross_val_score(self, X, y, cv=5):
        """Compute the cross-validation scores for a model."""
        scoring = ["accuracy", "precision", "recall"]
        cv_results = cross_validate(self.Pipeline, X, y,
                                    scoring=scoring, cv=cv, n_jobs=-1)
        print("Accuracy = {}".format(np.mean(cv_results["test_accuracy"])))
        print("Precision = {}".format(np.mean(cv_results["test_precision"])))
        print("Recall = {}".format(np.mean(cv_results["test_recall"])))

    def feature_importance(self, feature_names):
        """Compute and plot feature importances.

        features should be a list of strings of the feature names.
        """
        if self.scaling:
            f = self.Pipeline.steps[1][1].feature_importances_
        else:
            f = self.Pipeline.steps[0][1].feature_importances_
        f /= max(f)
        idx = np.argsort(f)
        feature_names = list(np.array(feature_names)[idx])
        plt.subplots(figsize=(10, 10))
        plt.title("Feature Importance")
        plt.barh(range(f.shape[0]), f[idx], color="b")
        plt.yticks(range(f.shape[0]), feature_names)
        plt.show()
