import pandas as pd
from transformers import pipeline


class FinBertSentimentAnalyzer:

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        # Initialize FinBERT sentiment pipeline
        self.classifier = pipeline(
            "sentiment-analysis", model=model_name, top_k=None
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Prepare text batch (truncated to 512 tokens max)
        texts = df["combined_text"].fillna("").astype(str).str[:512].tolist()
        print(f"[+] Running FinBERT inference on {len(texts)} posts...")

        results = self.classifier(texts)

        pos_scores, neg_scores, neu_scores = [], [], []

        for res in results:
            score_dict = {}

            # Case 1: List of prediction dictionaries
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        lbl = str(item.get("label", "")).lower()
                        val = float(item.get("score", 0.0))
                        score_dict[lbl] = val

            # Case 2: Single prediction dictionary
            elif isinstance(res, dict):
                lbl = str(res.get("label", "")).lower()
                val = float(res.get("score", 0.0))
                score_dict[lbl] = val

            pos_scores.append(score_dict.get("positive", 0.0))
            neg_scores.append(score_dict.get("negative", 0.0))
            neu_scores.append(score_dict.get("neutral", 0.0))

        df["pos_score"] = pos_scores
        df["neg_score"] = neg_scores
        df["neu_score"] = neu_scores

        # Net sentiment score: Positive minus Negative
        df["sentiment_score"] = df["pos_score"] - df["neg_score"]
        return df

    @staticmethod
    def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates daily post scores into a single daily net sentiment metric."""
        if df.empty:
            return pd.DataFrame()

        daily_df = (
            df.groupby("date")
            .agg(
                daily_net_sentiment=("sentiment_score", "mean"),
                total_posts=("post_id", "count"),
            )
            .reset_index()
        )
        return daily_df


# Alias for backward compatibility
SentimentEngine = FinBertSentimentAnalyzer