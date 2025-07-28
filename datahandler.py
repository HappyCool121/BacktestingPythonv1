# data collection and cleaning module

import pandas as pd
import yfinance as yf

# --- Module 1: Data Collection from yfinance ---
class DataHandler:
    """
    Handles fetching and preparing the financial data.
    """
    def __init__(self, symbol: str, start_date: str, end_date: str, interval: str = "1d"):
        """
        Initializes the DataHandler.
        Args:
            symbol (str): The ticker symbol to download (e.g., "BTC-USD").
            start_date (str): The start date for the data in "YYYY-MM-DD" format.
            end_date (str): The end date for the data in "YYYY-MM-DD" format.
            interval (str): The data interval (e.g., "1d", "1h").
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def get_data(self) -> pd.DataFrame:
        """
        Downloads data from yfinance, flattens the column structure,
        and prepares it with a simple header.

        Returns:
            pd.DataFrame: A DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume' columns.
                          Returns an empty DataFrame if download fails.
        """
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
        # Download data from yfinance
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.interval)

        if df.empty:
            print(f"Error: No data found for symbol {self.symbol}. Please check the ticker.")
            return pd.DataFrame()

        # 1. Reset the index to make the 'Date' index a regular column
        df.reset_index(inplace=True)

        # 2. Select the desired columns and rename them to the specified format.
        # This approach is robust and handles both single and multi-level column headers.
        # It also implicitly drops the 'Adj Close' column.
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        print("Data processed successfully. DataFrame head:")
        print(df.head())
        return df
