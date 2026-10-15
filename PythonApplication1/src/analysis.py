import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def merge_and_align_lags(
    daily_sentiment_df: pd.DataFrame, stock_df: pd.DataFrame
) -> pd.DataFrame:
  """Merges daily sentiment with stock returns, ensuring returns are calculated

  on historical prices before alignment.
  """
  if daily_sentiment_df.empty or stock_df.empty:
    return pd.DataFrame()

  daily_sentiment_df["date"] = pd.to_datetime(daily_sentiment_df["date"])
  stock_df["date"] = pd.to_datetime(stock_df["date"])

  stock_df = stock_df.sort_values("date").reset_index(drop=True)
  if "Close" in stock_df.columns and "daily_return" not in stock_df.columns:
    stock_df["daily_return"] = stock_df["Close"].pct_change().fillna(0.0)

  merged = pd.merge(
      daily_sentiment_df,
      stock_df[["date", "Close", "daily_return"]],
      on="date",
      how="left",
  )
  merged = merged.sort_values("date").reset_index(drop=True)

  merged["Close"] = merged["Close"].ffill()
  merged["daily_return"] = merged["daily_return"].fillna(0.0)

  merged["return_same_day"] = merged["daily_return"]
  merged["return_next_day"] = merged["daily_return"].shift(-1)

  return merged


def compute_correlations(aligned_df: pd.DataFrame) -> dict:
  """Calculates Pearson correlation between net sentiment and stock returns."""
  if aligned_df.empty or len(aligned_df) < 3:
    return {"same_day": np.nan, "lag_1": np.nan}

  same_day_df = aligned_df.dropna(
      subset=["daily_net_sentiment", "return_same_day"]
  )
  lag_1_df = aligned_df.dropna(
      subset=["daily_net_sentiment", "return_next_day"]
  )

  corr_same = (
      same_day_df["daily_net_sentiment"].corr(same_day_df["return_same_day"])
      if len(same_day_df) >= 3
      else np.nan
  )
  corr_lag1 = (
      lag_1_df["daily_net_sentiment"].corr(lag_1_df["return_next_day"])
      if len(lag_1_df) >= 3
      else np.nan
  )

  return {"same_day": corr_same, "lag_1": corr_lag1}


def plot_results(aligned_df: pd.DataFrame, ticker: str = "NVDA"):
  """Renders a clean dual-axis bar plot with perfectly aligned zero axes."""
  if aligned_df.empty:
    print("[!] Cannot plot results: aligned dataset is empty.")
    return

  aligned_df = aligned_df.sort_values("date").reset_index(drop=True)
  aligned_df["date_str"] = pd.to_datetime(aligned_df["date"]).dt.strftime(
      "%Y-%m-%d"
  )

  x = np.arange(len(aligned_df))
  width = 0.35

  fig, ax1 = plt.subplots(figsize=(10, 5))

  y1 = aligned_df["daily_net_sentiment"]
  y2 = aligned_df["return_same_day"] * 100

  # Left Axis: Sentiment Bars
  color1 = "#1f77b4"
  ax1.bar(
      x - width / 2,
      y1,
      width,
      label="Daily Net Sentiment",
      color=color1,
      alpha=0.8,
  )
  ax1.set_xlabel("Date", fontweight="bold")
  ax1.set_ylabel("Daily Net Sentiment", color=color1, fontweight="bold")
  ax1.tick_params(axis="y", labelcolor=color1)

  # Right Axis: Stock Return Bars
  ax2 = ax1.twinx()
  color2 = "#2ca02c"
  ax2.bar(
      x + width / 2,
      y2,
      width,
      label="Stock Return (%)",
      color=color2,
      alpha=0.8,
  )
  ax2.set_ylabel("Stock Return (%)", color=color2, fontweight="bold")
  ax2.tick_params(axis="y", labelcolor=color2)

  # 強制對齊左右兩軸的零點 (Zero-Axis Symmetry Alignment)
  max_abs_y1 = max(abs(y1.min()), abs(y1.max()), 0.01) * 1.25
  max_abs_y2 = max(abs(y2.min()), abs(y2.max()), 0.5) * 1.25

  ax1.set_ylim(-max_abs_y1, max_abs_y1)
  ax2.set_ylim(-max_abs_y2, max_abs_y2)

  ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

  ax1.set_xticks(x)
  ax1.set_xticklabels(aligned_df["date_str"], rotation=0)

  plt.title(
      f"{ticker} Reddit Net Sentiment vs. Stock Return",
      fontsize=12,
      fontweight="bold",
  )
  fig.tight_layout()
  plt.grid(True, linestyle=":", alpha=0.4, axis="y")
  plt.show()