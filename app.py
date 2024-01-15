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


st.title('📈Solar Condition Monitoring System')
st.write('This app uses ARIMA model to forecast the Data.')

database_url = 'https://xion-solar-9fc15-default-rtdb.firebaseio.com/'

def load_data():
    response = requests.get(database_url + ".json")

    if response.status_code == 200:
        data = response.json()
        nested_data = data.get('A00000001', {}).get('Voltage', {})
        df = pd.DataFrame(list(nested_data.items()), columns=['Timestamp', 'Current'])
        df = df[df['Timestamp'].str.match(r'\d{2}:\d{2}:\d{2}$')]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
        df = df.dropna(subset=['Current'])
        df.set_index('Timestamp', inplace=True)
        #df = df.resample('10T').mean()

        return df
    else:
        st.write("Failed to retrieve data from Firebase:", response.status_code)
        return None


image_url = "logo.png"
st.sidebar.image(image_url)
display_dataframe = st.sidebar.checkbox("Display Dataframe")
plot_dataframe = st.sidebar.checkbox("Plot Dataframe in Real-time")
run_forecast_button = st.sidebar.checkbox("Run ARIMA Forecast")

df = load_data()

num_forecast_steps = st.sidebar.number_input("Number of Future Steps to Forecast", min_value=1, max_value=1000, value=10)

if display_dataframe and df is not None:
    st.subheader("Displaying Dataframe")
    st.write(df)

if plot_dataframe and df is not None:
    st.subheader("Plotting dataset")
    chart_placeholder = st.empty()

    while plot_dataframe:
        df_filtered = load_data()

        if df_filtered is not None:
            rolling_avg = df_filtered['Current'].rolling(window=5).mean()
            chart_placeholder.line_chart(rolling_avg, use_container_width=True)


if run_forecast_button and df is not None:

    df = df.dropna(subset=['Current'])
    X = df['Current'].values


    size = int(len(X) * 0.8)
    train, test = X[:size], X[size:]
    history = [x for x in train]
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(1, 1, 1))
        try:
            model_fit = model.fit()
        except ConvergenceWarning:
            continue

        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    # Forecast future values
    for _ in range(num_forecast_steps):
        model = ARIMA(history, order=(5, 2, 1))
        try:
            model_fit = model.fit()
        except ConvergenceWarning:
            continue

        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)

    # Evaluate the model
    mse = mean_squared_error(test, predictions[:len(test)])
    rmse = np.sqrt(mse)
    st.subheader('Model Evaluation')
    st.write('Mean Squared Error:', mse)
    st.write('Root Mean Squared Error:', rmse)

    st.subheader('Forecast Plot')
    # Forecast Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(X, label='Historical Data')
    ax.plot(np.arange(size, size + len(test)), test, label='Test')
    ax.plot(np.arange(size, size + len(test) + num_forecast_steps), predictions, label='Forecast')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Current')
    ax.legend()
    st.pyplot(fig)
