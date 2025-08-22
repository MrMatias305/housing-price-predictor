import json
import os.path
import joblib
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from preprocessing import feature_groups, load_dataset
from train_model import train_pipeline, split_data
from validation import validate_model

# Set title
st.set_page_config(page_title='Housing price predictor')
st.title(":blue[🏠 Housing Price Predictor]")
st.markdown("<hr>", unsafe_allow_html=True)

# Load model
model_file = os.path.join('..', 'models', 'best_model_pipeline.pkl')
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = train_pipeline()

# Load validation metrics
val_file = os.path.join('..', 'results', 'reports', 'validation_metrics.json')
if not os.path.exists(val_file):
    validate_model()

with open(val_file, 'r') as f:
    metrics = json.load(f)


# Visualize predictions
def visualize_pred():
    X_train, X_test, y_train, y_test = split_data()

    # Predict on test set
    y_pred = model.predict(X_test)

    # Visualize predictions
    fig, ax = plt.subplots()
    ax.plot(y_test, y_pred, 'b.', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r-', linewidth=2)
    ax.set_title("Actual vs Predicted Housing Prices")
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.grid(True)

    with st.expander("Actual vs predicted housing prices"):
        # Plot
        st.header(":green[Actual vs Predicted Housing Prices]", divider=True)
        st.pyplot(fig)

        # Validation metrics
        st.header(":green[Validation metrics]", divider=True)
        col_names = ["MSE", "RMSE", "R²"]
        scores = [metrics["mse_scores"], metrics["rmse_scores"], metrics["r2_scores"]]
        mean_scores = [metrics["mean_mse"], metrics["mean_rmse"], metrics["mean_r2"]]

        for i, col in enumerate(st.columns(3)):
            col.subheader(f":blue[{col_names[i]} Scores]")
            for score in scores[i]:
                col.write(f"{score:,.2f}")
            col.subheader(f":blue[Mean {col_names[i]}]")
            col.write(f"{mean_scores[i]:,.2f}")


# Feature importances
def cal_feature_importances():
    new_cat_features = (model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .named_steps['encoder']
                        .get_feature_names_out(cat_features))

    all_features = num_features + list(new_cat_features)
    feature_importances = model.named_steps['rf'].feature_importances_

    # Display feature importances
    st.header(":green[Feature Importances]", divider='blue')
    st.info("Select number of features")
    selected_n_features = st.slider('', 0, len(all_features), 10)

    feature_importances_df = pd.DataFrame(
        {'Features': all_features,
         'Importances': feature_importances}
    ).sort_values(by='Importances', ascending=False)

    st.dataframe(feature_importances_df.head(selected_n_features))
    st.bar_chart(feature_importances_df.set_index('Features')
                 .head(selected_n_features))


# Get feature groups
num_features, cat_features = feature_groups()

visualize_pred()
cal_feature_importances()


# Predict new data
st.header(":green[Predict Price]", divider='blue')

input_data = {}

X, y = load_dataset()
for feature in num_features:
    min_value = X[feature].min()
    max_value = X[feature].max()
    mean_value = X[feature].mean()
    label = ' '.join(feature.split('_')).capitalize()
    input_data[feature] = st.slider(label, min_value, max_value, mean_value)

for feature in cat_features:
    label = ' '.join(feature.split('_')).capitalize()
    options = X[feature].unique()
    input_data[feature] = st.selectbox(label=label, options=options)

predict_btn = st.button("Predict")
if predict_btn:
    input_df = pd.DataFrame([input_data])
    pred_price = model.predict(input_df)
    st.success(f"## Predicted price: :green[${pred_price[0]:,.2f}]")
