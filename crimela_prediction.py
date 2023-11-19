import bz2
import pickle
import pickle as cPickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

data = decompress_pickle('crimela_pkl.pbz2')

 # Streamlit app
st.title('Crime Type Prediction')

# Sidebar for user input
st.sidebar.title('Input Parameters')

# List of areas
area_options = [
    "77th Street",
    "Pacific",
    "Southwest",
    "Hollywood",
    "Southeast",
    "Olympic",
    "Newton",
    "N Hollywood",
    "Wilshire",
    "Rampart",
    "West LA",
    "Northeast",
    "Van Nuys",
    "West Valley",
    "Harbor",
    "Topanga",
    "Devonshire",
    "Mission",
    "Hollenbeck",
    "Foothill"
]

# Create a Dropbox for selecting areas
selected_area = st.sidebar.selectbox('Select Area', area_options)

# Premises selection dropdown
premises_options = [
    'Commercial Areas',
    'Miscellaneous',
    'Public Spaces',
    'Residential Areas',
    'Specialty Locations',
    'Transportation & Parking',
    'Uncategorized'
]
selected_premise = st.sidebar.selectbox('Select Premise', premises_options)

# Select time, day and month
time = st.sidebar.number_input('Enter time', min_value=1, max_value=2359, step=1)
day = st.sidebar.number_input('Select Day', min_value=1, max_value=31, step=1)
month = st.sidebar.number_input('Select Month', min_value=1, max_value=12, step=1)

# Mapping of days of the week to numerical representation
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

selected_day_of_week = st.sidebar.selectbox('Select Day of the Week', list(day_mapping.keys()))

# Display selected day of the week
if selected_day_of_week:
    numerical_day = day_mapping[selected_day_of_week]

# Encode categorical variables to one-hot encoding
def encode_categorical(selected_value, options):
    return [1 if selected_value == opt else 0 for opt in options]

# Encode categorical variables
encoded_premises = encode_categorical(selected_premise, premises_options)
encoded_areas = encode_categorical(selected_area, area_options)

# Create the feature array
feature_array = [
    time,  # TIME OCC
    day,  # day
    month,  # month
    day_mapping[selected_day_of_week],  # Day_of_Week_Num
    *encoded_premises,  # Premises options
    *encoded_areas  # Premises options
]

# Convert the list to a numpy array if required
feature_array = np.array(feature_array)
reshaped_feature_array = feature_array.reshape(1, -1)  # Reshape to a 2D array with a single sample

# st.write('Feature Array:', feature_array)

# Button for prediction
if st.button('Predict Crime Type'):
    # Placeholder for model prediction
    prediction = data.predict(reshaped_feature_array)  # Replace this with actual prediction code
    # Define the reverse mapping dictionary
    reverse_mapping = {
    1: 'Violent Crimes',
    2: 'Property Crimes',
    3: 'Fraud and White-Collar Crimes',
    4: 'Sexual Offenses',
    5: 'Weapons and Firearm Offenses',
    6: 'Disturbance and Public Order',
    7: 'Miscellaneous and Other Crimes'
    }

    # Decode the predicted crime type
    predicted_crime_type = reverse_mapping.get(int(prediction), 'Unknown')
    st.write('Predicted Crime Type:', predicted_crime_type)