
import os
import streamlit as st
import pandas as pd
import joblib

# ======================================================
# LOAD THE TRAINED MODEL
# ======================================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model.joblib"
)

model_bundle = joblib.load(model_path)

model = model_bundle["model"]

classification_threshold = model_bundle[
    "classification_threshold"
]

# ======================================================
# STREAMLIT APPLICATION
# ======================================================

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🌴",
    layout="centered"
)

st.title("🌴 Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on their profile and preferences.

Enter the customer details below to generate a prediction.
""")

st.divider()

# ======================================================
# CUSTOMER INFORMATION
# ======================================================

st.subheader("Customer Profile")

age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=37,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

# ======================================================
# TRAVEL PREFERENCES
# ======================================================

st.subheader("Travel Preferences")

number_of_persons = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

number_of_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

number_of_trips = st.number_input(
    "Number of Trips Annually",
    min_value=0,
    max_value=30,
    value=3,
    step=1
)

preferred_property_star = st.selectbox(
    "Preferred Property Star Rating",
    [3, 4, 5]
)

passport = st.selectbox(
    "Does the Customer Have a Passport?",
    ["Yes", "No"]
)

own_car = st.selectbox(
    "Does the Customer Own a Car?",
    ["Yes", "No"]
)

monthly_income = st.number_input(
    "Monthly Income (INR)",
    min_value=0,
    max_value=200000,
    value=23000,
    step=1000
)

# ======================================================
# CREATE INPUT DATAFRAME
# ======================================================

input_data = pd.DataFrame([{

    "Age": age,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_persons,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income

}])

# ======================================================
# PREDICTION
# ======================================================

if st.button("Predict Purchase", type="primary"):

    prediction_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        prediction_probability >= classification_threshold
    )

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{prediction_probability * 100:.2f}%"
    )

    if prediction == 1:

        st.success(
            "The customer is predicted to purchase "
            "the Wellness Tourism Package."
        )

    else:

        st.info(
            "The customer is predicted not to purchase "
            "the Wellness Tourism Package."
        )

    st.caption(
        f"Classification threshold used: "
        f"{classification_threshold}"
    )
