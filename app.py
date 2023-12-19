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

    # Forecast for the next steps
    st.subheader('Forecast')
    steps = st.number_input('Enter the number of steps to forecast', 1, len(test), 10)
    forecast = model_fit.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean
    st.write('Forecasted current for the next', steps, 'steps:')
    st.write(forecast_values)

    # Evaluation
    mse = mean_squared_error(test['Current'][:steps], forecast_values)
    rmse = np.sqrt(mse)
    st.write('Mean Squared Error:', mse)
    st.write('Root Mean Squared Error:', rmse)

    # Plot forecast
    st.subheader('Forecast Plot')
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(train['Current'], label='Train')
    ax.plot(test['Current'], label='Test')
    ax.plot(forecast_values, label='Forecast')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Current')
    ax.legend()
    st.pyplot(fig)

    # Future Predictions
    st.subheader('Future Predictions')
    future_steps = st.number_input('Enter the number of future steps to predict', 1, 1000, 10)
    future_forecast_obj = model_fit.get_forecast(steps=future_steps)
    future_forecast = future_forecast_obj.predicted_mean
    st.write('Future forecast for the next', future_steps, 'steps:')
    st.write(future_forecast)


