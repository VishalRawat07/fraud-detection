import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fraud_detection_pipeline.pkl")
st.title("Fraud Detection Prediction App")
st.markdown('Please enter the transaction details and use the predict button.')
st.divider()

transaction_type = st.selectbox(
    'Transaction type', ['PAYMENT', 'TRANSFER', 'CASH_OUT'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
old_balance_orig = st.number_input(
    'Old balance sender', min_value=0.0, value=10000.0)
new_balance_orig = st.number_input(
    'New balance sender', min_value=0.0, value=9000.0)
old_balance_dest = st.number_input(
    'Old balance receiver', min_value=0.0, value=0.0)
new_balance_dest = st.number_input(
    'New balance receiver', min_value=0.0, value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest,
        'balanceDiffOrig': old_balance_orig - new_balance_orig - amount,  # ← added
        'balanceDiffDest': new_balance_dest - old_balance_dest - amount,  # ← added
    }])

    prediction = model.predict(input_data)[0]

    st.subheader(f'Prediction: {prediction}')

    if prediction == 1:
        st.error('This transaction can be fraud.')
    else:
        st.success('This transaction looks like it is not a fraud.')
