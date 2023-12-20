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
st.write('This app uses ARIMA model to forecast the Current.')

# Replace the URL with the actual URL of your Firebase Realtime Database
database_url = "https://solar-monitoring-system-7b380-default-rtdb.firebaseio.com/"

def load_data():
    response = requests.get(database_url + ".json")

    if response.status_code == 200:
        data = response.json()

        solar_panel_id = '0001'
        device_id = 'Current'  # Assuming the current data is under the 'Current' key

        # Extract the relevant nested data
        nested_data = data.get('solar_panel', {}).get(solar_panel_id, {}).get(device_id, {})

        # Convert the nested data to a dataframe
        df = pd.DataFrame(list(nested_data.items()), columns=['Timestamp', 'Current'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Current'] = pd.to_numeric(df['Current'], errors='coerce')  # Convert 'Current' to numeric
        df = df.dropna(subset=['Current'])  # Drop rows with NaN in 'Current'
        df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index

        # Resample to 5-minute intervals
        df_resampled = df.resample('10T').mean()

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
            rolling_avg = df_filtered['Current'].rolling(window=5).mean()

            # Clear the previous chart and update with the new data
            chart_placeholder.line_chart(rolling_avg, use_container_width=True)

    # ... (previous code)

    # Run ARIMA Forecast button logic
if run_forecast_button and df is not None:

        X = df['Current'].values

        # Split the data into training and testing sets (80% training, 20% testing)
        size = int(len(X) * 0.8)
        train, test = X[:size], X[size:]

        # Initialize a list to store the training data
        history = [x for x in train]

        # Initialize a list to store the predicted values
        predictions = []

        # Loop through the testing set to make predictions
        for t in range(len(test)):
            # Try different orders
            model = ARIMA(history, order=(5, 2, 1))
            try:
                model_fit = model.fit()
            except ConvergenceWarning:
                # Handle convergence warning, try different orders or methods
                continue

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            #st.write(f'Predicted: {yhat:.2f}, Expected: {obs:.2f}')

        # Evaluate the model
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        st.subheader('Model Evaluation')
        st.write('Mean Squared Error:', mse)
        st.write('Root Mean Squared Error:', rmse)

        # Plot the actual vs. predicted values
        st.subheader('Actual vs. Predicted Plot')
        plt.plot(test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        st.pyplot(plt)

        st.subheader('Forecast Plot')
        # Forecast Plot
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(X, label='Historical Data')
        ax.plot(np.arange(size, size + len(test)), test, label='Test')
        ax.plot(np.arange(size, size + len(test)), predictions, label='Forecast')
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Current')
        ax.legend()
        st.pyplot(fig)
