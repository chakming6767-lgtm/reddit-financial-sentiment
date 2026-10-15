import os
import pandas as pd

from src.analysis import compute_correlations, merge_and_align_lags, plot_results
from src.market_data import fetch_stock_data
from src.scraper import fetch_reddit_posts
from src.sentiment_engine import FinBertSentimentAnalyzer


def run_pipeline(ticker: str = "NVDA", subreddit: str = "stocks"):
    os.makedirs("data", exist_ok=True)

    # 1. Scrape Live Reddit Data
    raw_df = fetch_reddit_posts(query=ticker, subreddit=subreddit, pages=3)
    if raw_df.empty:
        print("[!] Pipeline aborted: No Reddit data fetched.")
        return

    # 2. Run FinBERT Sentiment Scoring
    analyzer = FinBertSentimentAnalyzer()
    scored_df = analyzer.analyze_dataframe(raw_df)
    daily_sentiment_df = analyzer.aggregate_daily_sentiment(scored_df)

    # 3. Fetch Stock Returns
    min_date = daily_sentiment_df["date"].min()
    max_date = daily_sentiment_df["date"].max()
    stock_df = fetch_stock_data(
        ticker=ticker, start_date=min_date, end_date=max_date
    )

    if stock_df.empty:
        print("[!] Pipeline completed up to sentiment aggregation.")
        print(daily_sentiment_df.head())
        return

    # 4. Align Lags & Analyze Correlation
    analysis_df = merge_and_align_lags(daily_sentiment_df, stock_df)
    correlations = compute_correlations(analysis_df)

    print("\n" + "=" * 45)
    print(f"  CORRELATION RESULTS FOR {ticker}")
    print("=" * 45)
    print(f"  Same-Day Correlation : {correlations.get('same_day', 'N/A')}")
    print(f"  Lag-1 Correlation    : {correlations.get('lag_1', 'N/A')}")
    print("=" * 45 + "\n")

    # 5. Export Results & Visualize (Accumulate past runs without losing historical days)
    csv_path = f"data/{ticker}_pipeline_output.csv"

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            # Ensure date formats match before combining
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            analysis_df["date"] = pd.to_datetime(analysis_df["date"])

            combined_df = (
                pd.concat([existing_df, analysis_df])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True)
            )
        except Exception as e:
            print(f"[!] Warning: Failed to load existing CSV ({e}). Overwriting file.")
            combined_df = analysis_df
    else:
        combined_df = analysis_df

    combined_df.to_csv(csv_path, index=False)
    print(f"[+] Final pipeline dataset saved to '{csv_path}'.")

    # Render Visualization
    if len(combined_df) > 1:
        plot_results(combined_df, ticker=ticker)


if __name__ == "__main__":
    run_pipeline(ticker="NVDA", subreddit="stocks")