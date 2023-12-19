from datetime import time as dt_time
import streamlit as st
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

st.title('Voltage Forecasting using ARIMA')
st.write('This app uses ARIMA model to forecast the Voltage.')

# Replace the URL with the actual URL of your Firebase Realtime Database
database_url = "https://solar-monitoring-system-7b380-default-rtdb.firebaseio.com/"

def load_data():
    response = requests.get(database_url + ".json")

    if response.status_code == 200:
        data = response.json()

        solar_panel_id = '0001'
        device_id = 'Current'

        # Extract the relevant nested data
        nested_data = data.get('solar_panel', {}).get(solar_panel_id, {}).get(device_id, {})

        # Convert the nested data to a dataframe
        df = pd.DataFrame(list(nested_data.items()), columns=['Timestamp', 'Current'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Current'] = pd.to_numeric(df['Current'], errors='coerce')  # Convert 'Current' to numeric
        df = df.dropna(subset=['Current'])  # Drop rows with NaN in 'Current'
        df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index

        return df
    else:
        st.write("Failed to retrieve data from Firebase:", response.status_code)
        return None

# Add a sidebar with options


image_url = "logo.png"
st.sidebar.image(image_url)
display_dataframe = st.sidebar.checkbox("Display Dataframe")
plot_dataframe = st.sidebar.checkbox("Plot Dataframe in Real-time")
run_forecast_button = st.sidebar.checkbox("Run ARIMA Forecast")

# Load data
df = load_data()

# Display dataframe if selected in the sidebar
if display_dataframe and df is not None:
    st.subheader("Displaying Dataframe")
    st.write(df)

# Plot dataframe in real-time with a rolling average if selected in the sidebar
if plot_dataframe and df is not None:
    st.subheader("Plotting dataset")

    # Create a placeholder for the chart
    chart_placeholder = st.line_chart(df['Current'])  # Use 'Current' column directly as it is now the index

    while True:
        # Reload data in a loop
        df = load_data()

        if df is not None:
            # Calculate a rolling average with a window size of 5
            rolling_avg = df['Current'].rolling(window=5).mean()

            # Update the chart with the rolling average
            chart_placeholder.line_chart(rolling_avg)

# Add ARIMA model fitting and forecasting section
if run_forecast_button:
    # Split the data into train and test sets
    st.subheader('Train and Test Sets')
    split = st.slider('Select the train-test split ratio', 0.5, 0.9, 0.8, 0.01)
    train_size = int(len(df) * split)
    train, test = df[0:train_size], df[train_size:len(df)]
    st.write('Train set size:', len(train))
    st.write('Test set size:', len(test))

    # Fit the ARIMA model
    st.subheader('ARIMA Model')
    p = st.number_input('Enter the order of AR term', 0, 10, 5)
    d = st.number_input('Enter the degree of differencing', 0, 10, 1)
    q = st.number_input('Enter the order of MA term', 0, 10, 0)
    model = sm.tsa.ARIMA(train['Current'], order=(p, d, q))
    model_fit = model.fit()
    st.write(model_fit.summary())




