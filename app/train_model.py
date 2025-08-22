import json
import os.path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import load_dataset, make_preprocessor


def split_data():
    X, y = load_dataset()
    return train_test_split(X, y, test_size=.2, random_state=42)


def train_pipeline():
    X_train, X_test, y_train, y_test = split_data()

    # Full pipeline
    model_pipeline = Pipeline([('preprocessor', make_preprocessor()),
                               ('rf', RandomForestRegressor(random_state=42))])

    # Fine-tuning (RandomizedSearchCV)
    # Hyperparameters
    params = {
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__max_features': ['sqrt', 'log2', None],
        'rf__bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(model_pipeline, params,
                                       n_jobs=-1, cv=3, n_iter=10,
                                       scoring='neg_mean_squared_error',
                                       random_state=42, return_train_score=True)

    random_search.fit(X_train, y_train)
    best_model_pipeline = random_search.best_estimator_
    model_info = random_search.best_params_

    # Save model
    model_path = os.path.join('..', 'models', 'best_model_pipeline.pkl')
    joblib.dump(best_model_pipeline, model_path)

    # Save model info
    model_info_path = os.path.join("..", "results", "reports", "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f)

    return best_model_pipeline
