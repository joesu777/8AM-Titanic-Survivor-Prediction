import streamlit as st
import pandas as pd
from fastai.tabular.all import *

st.title("Titanic Suvivorship Predictor")
st.caption("Built with Streamlit + FastAI")

model = load_learner("titanic_survivor_model.pkl")

def predict_survival(df):
    _, pred_idx, probs = model.predict(df.iloc[0])
    confidence = probs[pred_idx]
    return pred_idx, confidence

st.header("Passenger Information")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
title = st.selectbox("Title", model.dls.classes["Title"])
deck = st.selectbox("Cabin Deck", model.dls.classes["Deck"])
embarked = st.selectbox("Embarked", model.dls.classes["Embarked"])

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents / Children Aboard", min_value=0, value=0)

age_missing = st.checkbox("Age Unknown")
age_na = 0
if age_missing == True:
    age_na = 1
else:
    age_na = 0




input_data = pd.DataFrame([{
    "Pclass":pclass,
    "Sex":sex,
    "Embarked": embarked,
    "Title": title,
    "Deck": deck,
    "Age_na":age_na,
    "Age":age,
    "SibSp":sibsp,
    "Parch":parch,
    "Fare":fare
}])

st.write("Categorical columns:", model.dls.cat_names)
st.write("Continuous columns:", model.dls.cont_names)
st.write("Input columns:", input_data.columns.tolist())

if st.button("Predict Survival"):
    pred_idx, confidence = predict_survival(input_data)
    if pred_idx == 1:
        st.success(f"Survived (confidence: {confidence:.2f})")
    else:
        st.error(f"Did Not Survive (confidence: {confidence:.2f})")


