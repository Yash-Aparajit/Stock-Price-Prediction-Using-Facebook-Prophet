# Installing the necessary library
pip install prophet

# Importing required libraries
import pandas as pd
import plotly.express as px
from prophet import Prophet
import yfinance as yf
import plotly.io as pio

# Setting Plotly's default renderer for Google Colab
pio.renderers.default = 'colab'

# Download Zomato Stock Data from Yahoo Finance (ZOMATO.BO) for a specific time period
ZOMATO_BO_data = yf.download("ZOMATO.BO", start="2023-04-23", end="2025-04-23")

# Save the data into a CSV file
ZOMATO_BO_data.to_csv("ZOMATO.BO.csv")

# Load the CSV data into a pandas DataFrame
zomato_df = pd.read_csv("ZOMATO.BO.csv")

# Display the basic structure of the DataFrame
df = zomato_df
df.info()

# Rename the 'Price' column to 'Date' for clarity
df.rename(columns={'Price': 'Date'}, inplace=True)

# Convert columns with numeric data to proper numeric types
cols_to_fix = ['Open', 'Close', 'High', 'Low', 'Volume']
df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

# Display statistical summary of the DataFrame
df.describe()

# Plotting the closing stock price as an area chart
px.area(df, x="Date", y="Close", title="Zomato Stock (Closing)")

# Plotting the stock volume as an area chart
px.area(df, x="Date", y="Volume", title="Zomato Stock (Volume)")

# Creating a boxplot of the stock closing prices
px.box(df, y="Close")

# Selecting only the necessary columns for Prophet (Date and Close)
columns = ['Date', "Close"]
ndf = pd.DataFrame(df, columns=columns)

# Prophet requires 'ds' for the date column and 'y' for the value column
prophet_df = ndf.rename(columns={'Date': 'ds', 'Close': 'y'})

# Convert the 'ds' column to a datetime format, coerce any errors, and drop rows with missing dates
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format="%d/%m/%Y", errors='coerce')
prophet_df = prophet_df.dropna(subset=['ds'])

# Instantiate the Prophet model
m = Prophet()

# Fit the model with the data
m.fit(prophet_df)

# Generate a future dataframe for the next 30 days 
future = m.make_future_dataframe(periods=30)

# Predict the stock prices using the fitted model
forecast = m.predict(future)

# Plotting the predicted stock prices using Plotly
px.line(forecast, x="ds", y="yhat", title="Zomato Stock Price Prediction")

# Plot the forecast using Prophet's built-in plotting function
figure = m.plot(forecast, xlabel='ds', ylabel='y')

# Plotting the Forecast Components (Trend, Seasonality)
figure2 = m.plot_components(forecast)

# Save the forecasted data to a CSV file
from google.colab import files
forecast.to_csv('forecast.csv')

# Provide the CSV file for download
files.download('forecast.csv')
