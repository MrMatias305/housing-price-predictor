import os.path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset():
    # Load data
    data = pd.read_csv(os.path.join("..", "data", "housing.csv"))
    data.columns = [" ".join(col.split("_")).capitalize() for col in data.columns]

    X = data.drop(columns='Median house value')
    y = data['Median house value']

    # Modify columns
    X['Rooms per household'] = data['Total rooms'] / data['Households']
    X['Bedrooms per rooms'] = data['Total bedrooms'] / data['Total rooms']
    X['Population per household'] = data['Population'] / data['Households']
    X['Median income'] = data['Median income'] * 10_000
    X = X.drop(columns=['Total rooms', 'Total bedrooms', 'Population', 'Households'])

    return X, y


def feature_groups():
    X, y = load_dataset()
    numerical_features = [col for col in X.columns if col != 'Ocean proximity']
    categorical_features = ['Ocean proximity']
    return numerical_features, categorical_features


def make_preprocessor():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    # Get feature groups
    num_features, cat_features = feature_groups()

    # Preprocessor
    preprocessor = ColumnTransformer(
        [('num', num_pipeline, num_features),
         ('cat', cat_pipeline, cat_features)],
        n_jobs=-1)

    return preprocessor
