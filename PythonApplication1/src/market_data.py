import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker: str = "NVDA", start_date: str = None, end_date: str = None
) -> pd.DataFrame:
  """Fetches historical stock prices via yfinance and ensures clean date formatting."""
  try:
    end_dt = pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days=45)  # 確保足夠涵蓋歷史範圍

    s_str = start_dt.strftime("%Y-%m-%d")
    e_str = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[+] Downloading stock market data for {ticker} from {s_str} to {e_str}...")
    df = yf.download(ticker, start=s_str, end=e_str, progress=False)

    if df.empty:
      print(f"[!] Warning: No price data returned for ticker {ticker}.")
      return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Date" in df.columns:
      df.rename(columns={"Date": "date"}, inplace=True)

    # 統一轉成字串格式以防 Pandas merge 失敗
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    if "Close" in df.columns:
      df["daily_return"] = df["Close"].pct_change()
    else:
      df["daily_return"] = 0.0

    return df[["date", "Close", "daily_return"]].dropna(subset=["daily_return"])

  except Exception as e:
    print(f"[!] Error fetching stock data: {e}")
    return pd.DataFrame()