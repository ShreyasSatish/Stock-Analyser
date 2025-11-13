# Import statements - run: pip install -r requirements2.txt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
from neuralprophet import NeuralProphet, set_log_level
from collections import OrderedDict
from neuralprophet.configure import (
    ConfigSeasonality,
    ConfigFutureRegressors,
    ConfigLaggedRegressors,
    ConfigEvents,
    ConfigCountryHolidays,
    Season,
    Trend
)
import warnings

warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals([
    ConfigSeasonality,
    ConfigFutureRegressors,
    ConfigLaggedRegressors,
    ConfigEvents,
    ConfigCountryHolidays,
    Season,
    Trend,
    OrderedDict
])


"""Testing Neural Prohpet module first"""
df = yf.download("AAPL", start="2016-01-01", end="2024-01-01",
                 auto_adjust=False, multi_level_index=False)
df = df["Close"] # What we are trying to predict
df = df.reset_index()
df = df.rename(columns={"Date":"ds", "Close":"y"})
# Neural Prophet only takes in two columns, ds for datestamp 
# and y for the column we are predicting
print(df.head())

# Plotting
plt.figure(figsize=(10,6))
plt.plot(df["ds"], df["y"])
plt.show()

set_log_level("ERROR") # Turns off logging messages unless there is an error in the code
m = NeuralProphet()
m.set_plotting_backend("plotly-static")
metrics = m.fit(df)
# Create a new df that goes 365 days into future for forecast
df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=365)
forecast = m.predict(df_future) # Store prediction
m.plot(forecast)

