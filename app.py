from datetime import time

import streamlit as st
import pandas as pd
import requests

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
        df = df.dropna()  # Drop rows with NaT (Not a Time) values
        df.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index

        return df
    else:
        st.write("Failed to retrieve data from Firebase:", response.status_code)
        return None

# Add a sidebar with options
st.sidebar.title("Options")
display_dataframe = st.sidebar.checkbox("Display Dataframe")
plot_dataframe = st.sidebar.checkbox("Plot Dataframe in Real-time")

# Load data
df = load_data()

# Display dataframe if selected in the sidebar
if display_dataframe and df is not None:
    st.subheader("Displaying Dataframe")
    st.write(df)

# Plot dataframe in real-time with a rolling average if selected in the sidebar
if plot_dataframe and df is not None:
    st.subheader("Plotting Dataframe in Real-time with Rolling Average")

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
