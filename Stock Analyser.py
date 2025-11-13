# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

class StockAnalyser():

    def __init__(self, stock="AAPL", 
                 start="2021-01-01", 
                 end="2025-01-01"):
        self.stock = stock
        self.start = start
        self.end = end

        self.df = yf.download(self.stock, self.start, self.end) 
        # Remove null and na values
        self.df.isnull().values.any()
        self.df = self.df.dropna()
    

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
        plt.figure(figsize=(8,6))
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

        if visualise:
            # Make figure and plot data
            plt.figure(figsize=(8,6))
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


def main():
    analyse = StockAnalyser()
    print(analyse.head())
    # analyse.lag_plot(5)
    analyse.train_test_split(column="High")


if __name__ == "__main__":
    main()