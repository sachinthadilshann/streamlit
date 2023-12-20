from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

st.title('Current Forecasting using ARIMA')
st.text("Welcome to the Current Forecasting App using ARIMA. Select options in the sidebar to explore the data and run forecasts.")

# Replace the URL with the actual URL of your Firebase Realtime Database
database_url = "https://solar-monitoring-system-7b380-default-rtdb.firebaseio.com/"

def load_data():
    with st.spinner('Loading data...'):
        response = requests.get(database_url + ".json")

        if response.status_code == 200:
            data = response.json()

            solar_panel_id = '0001'
            device_id = 'Current'
            
            nested_data = data.get('solar_panel', {}).get(solar_panel_id, {}).get(device_id, {})

            df = pd.DataFrame(list(nested_data.items()), columns=['Timestamp', 'Current'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
            df = df.dropna(subset=['Current'])
            df.set_index('Timestamp', inplace=True)

            df_resampled = df.resample('10T').mean()

            return df_resampled
        else:
            st.write("Failed to retrieve data from Firebase:", response.status_code)
            return None

# Sidebar Options
display_dataframe = st.sidebar.button("Display Dataframe")
plot_dataframe = st.sidebar.button("Plot Dataframe in Real-time")
run_forecast_button = st.sidebar.button("Run ARIMA Forecast")

# Create a multi-column layout
col1, col2 = st.columns(2)

# Load data outside the loop to avoid unnecessary data fetching
df = load_data()

# Display dataframe if selected in the sidebar
if display_dataframe and df is not None:
    col1.subheader("Displaying Dataframe")
    col1.write(df)

# Plot dataframe in real-time with a rolling average if selected in the sidebar
if plot_dataframe and df is not None:
    col2.subheader("Plotting dataset")

    # Create a placeholder for the chart
    chart_placeholder = col2.empty()

    while plot_dataframe:
        df_filtered = load_data()

        if df_filtered is not None:
            rolling_avg = df_filtered['Current'].rolling(window=5).mean()
            chart_placeholder.line_chart(rolling_avg, use_container_width=True)

    plt.close()  # Close the plot to avoid overlapping

# Run ARIMA Forecast button logic
if run_forecast_button and df is not None:
    X = df['Current'].values
    size = int(len(X) * 0.8)
    train, test = X[:size], X[size:]
    history = [x for x in train]
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 2, 1))
        try:
            model_fit = model.fit()
        except ConvergenceWarning:
            st.warning("Model convergence warning. Try different orders or methods.")
            continue

        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    
    st.subheader('Model Evaluation')
    st.write('Mean Squared Error:', mse)
    st.write('Root Mean Squared Error:', rmse)

    # Actual vs. Predicted Plot
    plt.plot(test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.legend()
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid overlapping

    # Forecast Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(X, label='Historical Data')
    ax.plot(np.arange(size, size + len(test)), test, label='Test')
    ax.plot(np.arange(size, size + len(test)), predictions, label='Forecast')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Current')
    ax.legend()
    st.pyplot(fig)
    plt.close()  # Close the plot
