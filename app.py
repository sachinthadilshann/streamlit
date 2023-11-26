import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title('Voltage Forecasting using ARIMA')
st.write('This app uses ARIMA model to forecast the Voltage .')
data = st.file_uploader('Upload a CSV file(Data,Voltage) only ', type='csv')

if data is not None:
    df = pd.read_csv(data, parse_dates=['Date'], index_col='Date')
    st.write(df.head())

    st.subheader('Voltage Plot')
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(df['Voltage'], label='Voltage')
    ax.set_xlabel('Date')
    ax.set_ylabel('Voltage')
    ax.legend()
    st.pyplot(fig)

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
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    st.write(model_fit.summary())

    st.subheader('Future Predictions')
    future_steps = st.number_input('Enter the number of future steps to predict', 1, 1000, 10)
    future_forecast_obj = model_fit.get_forecast(future_steps, alpha=0.05)
    future_forecast = future_forecast_obj.predicted_mean
    future_conf = future_forecast_obj.conf_int()
    st.write('Future forecast for the next', future_steps, 'hours:')
    st.write(future_forecast)

    # Plot the forecast
    st.subheader('Forecast Plot')
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df['Voltage'], label='Historical Data')
    ax.plot(future_forecast, label='Future Forecast')
    ax.fill_between(future_conf.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1], color='k', alpha=.15, label='Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Voltage')
    ax.legend()
    st.pyplot(fig)

    # Update the forecast in real-time
    st.subheader('Real-time Forecast Update')
    update_button = st.button('Update Forecast')
    if update_button:
        # Refit the model with the entire dataset
        model = ARIMA(df, order=(p, d, q))
        model_fit = model.fit()

        # Get the new forecast
        future_forecast_obj = model_fit.get_forecast(future_steps, alpha=0.05)
        future_forecast = future_forecast_obj.predicted_mean
        future_conf = future_forecast_obj.conf_int()

        # Update the plot
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(df['Voltage'], label='Historical Data')
        ax.plot(future_forecast, label='Future Forecast')
        ax.fill_between(future_conf.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1], color='k', alpha=.15, label='Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Voltage')
        ax.legend()
        st.pyplot(fig)
