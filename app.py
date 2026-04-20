import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and label encoders
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError:
    st.error("Model or label encoders not found. Please ensure 'linear_regression_model.pkl' and 'label_encoders.pkl' are in the same directory.")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the details to predict the salary.')

# Helper function to get original labels from encoder
def get_original_labels(encoder):
    if hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    return []

# Create input fields for each feature
# Assuming the order of features as in X: 'Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'

rating = st.slider('Rating', min_value=0.0, max_value=5.0, value=3.5, step=0.1)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5, step=1)

company_name_options = get_original_labels(label_encoders.get('Company Name', None))
if company_name_options:
    company_name_display = st.selectbox('Company Name', options=company_name_options)
    company_name_encoded = label_encoders['Company Name'].transform([company_name_display])[0]
else:
    st.warning("Company Name encoder not found or has no classes. Please ensure label_encoders.pkl is correct.")
    company_name_encoded = st.number_input('Company Name (encoded)', value=0, help="Enter the numerical encoding if original labels are not available.")

job_title_options = get_original_labels(label_encoders.get('Job Title', None))
if job_title_options:
    job_title_display = st.selectbox('Job Title', options=job_title_options)
    job_title_encoded = label_encoders['Job Title'].transform([job_title_display])[0]
else:
    st.warning("Job Title encoder not found or has no classes. Please ensure label_encoders.pkl is correct.")
    job_title_encoded = st.number_input('Job Title (encoded)', value=0, help="Enter the numerical encoding if original labels are not available.")

location_options = get_original_labels(label_encoders.get('Location', None))
if location_options:
    location_display = st.selectbox('Location', options=location_options)
    location_encoded = label_encoders['Location'].transform([location_display])[0]
else:
    st.warning("Location encoder not found or has no classes. Please ensure label_encoders.pkl is correct.")
    location_encoded = st.number_input('Location (encoded)', value=0, help="Enter the numerical encoding if original labels are not available.")

employment_status_options = get_original_labels(label_encoders.get('Employment Status', None))
if employment_status_options:
    employment_status_display = st.selectbox('Employment Status', options=employment_status_options)
    employment_status_encoded = label_encoders['Employment Status'].transform([employment_status_display])[0]
else:
    st.warning("Employment Status encoder not found or has no classes. Please ensure label_encoders.pkl is correct.")
    employment_status_encoded = st.number_input('Employment Status (encoded)', value=0, help="Enter the numerical encoding if original labels are not available.")

job_roles_options = get_original_labels(label_encoders.get('Job Roles', None))
if job_roles_options:
    job_roles_display = st.selectbox('Job Roles', options=job_roles_options)
    job_roles_encoded = label_encoders['Job Roles'].transform([job_roles_display])[0]
else:
    st.warning("Job Roles encoder not found or has no classes. Please ensure label_encoders.pkl is correct.")
    job_roles_encoded = st.number_input('Job Roles (encoded)', value=0, help="Enter the numerical encoding if original labels are not available.")


if st.button('Predict Salary'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame([[
        rating,
        company_name_encoded,
        job_title_encoded,
        salaries_reported,
        location_encoded,
        employment_status_encoded,
        job_roles_encoded
    ]],
    columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported',
             'Location', 'Employment Status', 'Job Roles'])

    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ₹{prediction:,.2f}')
