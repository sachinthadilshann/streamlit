from datetime import datetime
from datetime import time as dt_time
import streamlit as st
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA from statsmodels

st.title('Voltage Forecasting using ARIMA')
st.write('This app uses ARIMA model to forecast the Voltage.')

# Replace the URL with the actual URL of your Firebase Realtime Database
database_url = "https://solar-monitoring-system-7b380-default-rtdb.firebaseio.com/"

def load_data():
    response = requests.get(database_url + ".json")

    if response.status_code == 200:
        data = response.json()

        solar_panel_id = '0001'
        device_id = 'Voltage'  # Assuming the voltage data is under the 'Voltage' key

        # Extract the relevant nested data
        nested_data = data.get('solar_panel', {}).get(solar_panel_id, {}).get(device_id, {})

        # Convert the nested data to a dataframe
        df = pd.DataFrame(list(nested_data.items()), columns=['Timestamp', 'Voltage'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')  # Convert 'Voltage' to numeric
        df = df.dropna(subset=['Voltage'])  # Drop rows with NaN in 'Voltage'
        df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index

        # Resample to 5-minute intervals
        df_resampled = df.resample('30S').mean()

        return df_resampled
    else:
        st.write("Failed to retrieve data from Firebase:", response.status_code)
        return None

# Add a sidebar with options
image_url = "logo.png"
st.sidebar.image(image_url)
display_dataframe = st.sidebar.checkbox("Display Dataframe")
plot_dataframe = st.sidebar.checkbox("Plot Dataframe in Real-time")
run_forecast_button = st.sidebar.checkbox("Run ARIMA Forecast")

# Load data outside the loop to avoid unnecessary data fetching
df = load_data()

# Display dataframe if selected in the sidebar
if display_dataframe and df is not None:
    st.subheader("Displaying Dataframe")

    # Display the filtered dataframe
    st.write(df)

# Plot dataframe in real-time with a rolling average if selected in the sidebar
if plot_dataframe and df is not None:
    st.subheader("Plotting dataset")

    # Create a placeholder for the chart
    chart_placeholder = st.empty()

    while plot_dataframe:  # Continue plotting as long as the checkbox is selected
        # Reload data in a loop
        df_filtered = load_data()

        if df_filtered is not None:
            # Calculate a rolling average with a window size of 5
            rolling_avg = df_filtered['Voltage'].rolling(window=5).mean()

            # Clear the previous chart and update with the new data
            chart_placeholder.line_chart(rolling_avg, use_container_width=True)

# Run ARIMA Forecast button logic
if run_forecast_button and df is not None:
    # Split the data into train and test sets
    st.subheader('Train and Test Sets')
    split = st.slider('Select the train-test split ratio', 0.5, 0.9, 0.8, 0.01)
    train_size = int(len(df) * split)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Fit the ARIMA model
    st.subheader('ARIMA Model')
    p = st.number_input('Enter the order of AR term', 0, 10, 5)
    d = st.number_input('Enter the degree of differencing', 0, 10, 1)
    q = st.number_input('Enter the order of MA term', 0, 10, 0)

    # Create and fit the ARIMA model
    model = ARIMA(train['Voltage'], order=(p, d, q))
    model_fit = model.fit()
    st.write(model_fit.summary())

    # Forecast
    st.subheader('Forecast')
    steps = st.number_input('Enter the number of steps to forecast', 1, len(test), 10)
    forecast_obj = model_fit.get_forecast(steps, alpha=0.05)
    forecast = forecast_obj.predicted_mean
    se = forecast_obj.se_mean
    conf = forecast_obj.conf_int()
    st.write('Forecasted voltage for the next', steps, 'steps:')
    st.write(forecast)

    # Evaluation
    st.subheader('Evaluation')
    mse = mean_squared_error(test['Voltage'][:steps], forecast)
    rmse = np.sqrt(mse)
    st.write('Mean Squared Error:', mse)
    st.write('Root Mean Squared Error:', rmse)

    # Forecast Plot
    st.subheader('Forecast Plot')
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(train['Voltage'], label='Train')
    ax.plot(test['Voltage'], label='Test')
    ax.plot(forecast.index, forecast.values, label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Voltage')
    ax.legend()
    st.pyplot(fig)

# ... (rest of the code)
