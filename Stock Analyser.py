# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore")

class StockAnalyser():

    def __init__(self, stock="AAPL", 
                 start="2015-01-01", 
                 end="2024-01-01"):
        self.stock = stock
        self.start = start
        self.end = end

        self.df = yf.download(self.stock, self.start, self.end) 
        # Remove null and na values
        self.df.isnull().values.any()
        self.df = self.df.dropna()
    
    # # Define a loss function
    # # in addition to the one imported 
    # # via sklearn
    # # Symmetric Moving Average Percentage Error
    # def smape(y_true, y_pred, epsilon=1e-8):
    #     # Calculate denominator and add epsilon
    #     # to avoid division by zero
    #     denom = (np.abs(y_pred) + np.abs(y_true)) + epsilon
    #     # Calculate absolute percentage error with 
    #     # symmetric scaling
    #     ape = np.abs(y_pred - y_true) * 200 / denom
    #     # Calculate the mean of smape
    #     mean_smape = np.mean(ape)
    #     return mean_smape

    def head(self, n=5):
        # Define a .head method so the user can get 
        # a view of the loaded data
        return self.df.head(n)
    
    def lag_plot(self, n=1):
        # Make a n day lag plot
        # Make a copy of the data so as to not
        # change the oirignal in an undesired way

        data = self.df.copy()
        data["Lagged Close"] = data["Close"].shift(n)
        data = data.dropna() # Drop any na values made by .shift(n)

        # Make figure and plot data
        plt.figure(figsize=(10,6))
        plt.scatter(x=data["Close"],
                    y=data["Lagged Close"],
                    marker="o", s=15
                    )
        plt.title(f"Lag plot of {self.stock} Closing Prices with a {n}-Day Lag")
        plt.xlabel("Todays Close")
        plt.ylabel(f"Close {n} Days ago")
        plt.grid(True)
        plt.show()

    def train_test_split(self, train=0.8, visualise=True, 
                         train_colour="blue", test_colour="green",
                         column="Open"):
        # Split the data
        self.train_data = self.df.iloc[0:int(len(self.df)*train), :]
        self.test_data = self.df.iloc[int(len(self.df)*train):, :]

        # Plot train and test if True
        if visualise:
            # Make figure and plot data
            plt.figure(figsize=(10,6))
            plt.scatter(x=self.train_data.index,
                        y=self.train_data[column],
                        label="Training Data",
                        color=train_colour
                        )
            plt.scatter(x=self.test_data.index,
                        y=self.test_data[column],
                        label="Testing Data",
                        color=test_colour
                        )
            plt.title(f"{self.stock} {column} Prices, Training and Testing Data")
            plt.xlabel("Dates")
            plt.ylabel(f"{column} Prices")
            plt.legend()
            plt.grid(True)
            plt.show()

    def rolling_stats(self, window=7, column="Open", 
                      visualise=True, orig_colour="blue",
                      mean_colour="red", std_colour="green"):
        self.train_series = self.train_data[column]

        # Determine rolling stats mean and std
        self.rolmean = self.train_series.rolling(window).mean()
        self.rolstd = self.train_series.rolling(window).std()

        # Plot statistics if True
        if visualise:
            # Make figure and plot data
            plt.figure(figsize=(10,6))
            # Plot the original first
            plt.plot(self.train_series, color=orig_colour, 
                     label="Original Data")
            # Then plot the mean
            plt.plot(self.rolmean, color=mean_colour, 
                     label="Rolling Mean")
            # Finally plot the std
            plt.plot(self.rolstd, color=std_colour, 
                     label="Rolling Standard Deviation")
            plt.legend()
            plt.grid(True)
            plt.title(f"Rolling Mean & Standard Deviation for {self.stock}")
            plt.xlabel("Dates")
            plt.show()

    def adfuller_test(self, column="Open", autolag="AIC"):
        # Run an Augmented Dickey Fuller Test
        self.train_series = self.train_data[column]
        dftest = adfuller(self.train_series, autolag=autolag)
        self.dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value", 
                                                 "Lags Used", 
                                                 "Number of Observations Used"])
        for key, value in dftest[4].items():
            self.dfoutput["Critical Value (%s)"%key] = value
        print("==== Undifferenced Stats ====")
        print(self.dfoutput)

    def make_stationary(self, periods=1, run_adfuller=True, 
                        run_rol_stats=True, autolag="AIC",
                        window=7):
        # Run differencing to make data more stationary
        # This is to only be used after using the adfuller_test
        # method defined above
        self.train_diff = self.train_series.diff(periods=periods)
        # Remove any nan values
        self.train_diff = self.train_diff.dropna()

        # Calculate and plot if True
        if run_rol_stats:
            # Determine rolling mean and std
            self.rolmean = self.train_diff.rolling(window).mean()
            self.rolstd = self.train_diff.rolling(window).std()

            # Plot stats
            # Make figure and plot data
            plt.figure(figsize=(10,6))
            # Plot the differenced data first
            plt.plot(self.train_diff, color="blue", 
                     label="Differenced Data")
            # Then plot the mean
            plt.plot(self.rolmean, color="red", 
                     label="Rolling Mean")
            # Finally plot the std
            plt.plot(self.rolstd, color="green", 
                     label="Rolling Standard Deviation")
            plt.legend()
            plt.grid(True)
            plt.title(f"Rolling Mean & Standard Deviation for Differenced {self.stock}")
            plt.xlabel("Dates")
            plt.show()

        if run_adfuller:
            dftest = adfuller(self.train_diff, autolag=autolag)
            self.dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value",
                                                          "Lags Used", 
                                                          "Number of Observations Used"])
            for key, value in dftest[4].items():
                self.dfoutput["Critical Value (%s)"%key] = value
            print("==== Differenced Stats ====")
            print(self.dfoutput)

    def run_ARIMA(self, periods=1, column="Open", 
                    p_val=5, d_val=1, q_val=0):
        # First perform differencing on the test data
        self.test_series = self.test_data[column]
        self.test_diff = self.test_series.diff(periods=periods)
        self.test_diff = self.test_diff.dropna()

        # Initalise history with training data
        history = self.train_diff.values.tolist()
        predictions = list()

        # Iterate through test data points
        for t in range(len(self.test_diff)):
            p, d, q = p_val, d_val, q_val
            model = ARIMA(np.array(history), order=(p,d,q))
            model_fit = model.fit()

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = self.test_diff.iloc[t]
            history.append(obs)

            if t % 100 == 0:
                clean_obs_value = obs.item()
                print(f"Test Series Point: {t}\tPredicted={yhat:.14f}, Expected={clean_obs_value}")
            
        # Calculate error using MSE
        error = mse(self.test_diff, predictions)
        print(f"Mean Squared Error (MSE): {error:.4f}")

def main():
    analyse = StockAnalyser()
    print(analyse.head())
    analyse.lag_plot(5)
    analyse.train_test_split(column="High")
    analyse.rolling_stats()
    analyse.adfuller_test(column="Close", autolag="BIC")
    analyse.make_stationary()
    analyse.run_ARIMA()


if __name__ == "__main__":
    main()