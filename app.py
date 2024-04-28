import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model = joblib.load('best_model.sav')

# Define feature columns
features = [
    "Transaction Amount", "Transaction Hour", "Transaction Day of Week",
    "Cardholder Name", "Merchant Category Code (MCC)",
    "Transaction Location (City or ZIP Code)", "Transaction Currency",
    "Card Type", "Previous Transactions", "Transaction Source",
    "IP Address", "Device Information"
]

st.title('Fraud Detection App')

# Input fields for user data
st.subheader('Enter User Data')
user_input = {}
for feature in features:
    user_input[feature] = st.text_input(f'{feature}', '')

# Add a button to trigger prediction
if st.button('Predict'):
    # Preprocess user input
    user_input_df = pd.DataFrame([user_input])
    for feature in features:
        if user_input_df[feature].dtype == 'object':
            le = LabelEncoder()
            user_input_df[feature] = le.fit_transform(user_input_df[feature])

    # Scale features
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input_df)

    # Make predictions
    prediction = model.predict(user_input_scaled)

    # Display prediction
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Fraud Predicted')
    else:
        st.write('Fraud Not Predicted')
