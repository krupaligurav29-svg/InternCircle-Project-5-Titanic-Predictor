import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Feature order must match Project 4
FEATURES = ["pclass", "sex", "age", "fare"]

# Load model only once
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# Application title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details below to estimate "
    "the passenger's survival probability."
)

st.divider()

# User inputs
st.subheader("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3],
    format_func=lambda x: {
        1: "1st Class",
        2: "2nd Class",
        3: "3rd Class"
    }[x]
)

sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0,
    max_value=500,
    value=30
)

st.divider()

# Prediction button
if st.button("Predict Survival", type="primary"):

    try:
        # Load saved model
        model = load_model()

        # Convert gender into the same encoding as Project 4
        sex_value = 1 if sex == "Female" else 0

        # Create input DataFrame in the correct feature order
        input_data = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        # Validate input values
        if input_data.isnull().any().any():
            st.error("Please provide valid passenger details.")
            st.stop()

        # Predict survival probability
        probability = model.predict_proba(input_data)[0][1]

        # Display result
        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Chance of Survival",
            value=f"{probability:.1%}"
        )

        if probability >= 0.5:
            st.success("The model predicts: Survived")
        else:
            st.warning("The model predicts: Did Not Survive")

    except FileNotFoundError:
        st.error(
            "The model file is missing. "
            "Please add model.joblib to the project folder."
        )

    except Exception as e:
        st.error(
            "Unable to generate the prediction. "
            "Please check the model and input values."
        )

# Educational note
st.divider()

st.caption(
    "This application uses a Decision Tree model trained on "
    "historical Titanic passenger data. It is an educational "
    "example and should not be treated as advice or a reliable "
    "prediction of real-world outcomes."
)