

# 📈 Reddit Financial Sentiment & Stock Correlation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Portfolio Project](https://img.shields.io/badge/Status-Active-success.svg)]()

An end-to-end, lightweight data science pipeline that tracks retail investor sentiment from Reddit financial discussions via RSS feeds, processes unstructured text using a hybrid **FinBERT + VADER** NLP engine, and evaluates statistical correlation against asset price movements using `yfinance`.

---

## 🚀 Key Engineering & Data Science Highlights

* **Lightweight & Efficient Data Extraction (No API Restrictions):** 
  Instead of depending on heavy API rate limits or complex OAuth credentials, this pipeline leverages **Reddit RSS feeds** for real-time public data collection, showcasing practical problem-solving under constrained environments.
* **Domain-Specific NLP Engine (FinBERT & VADER):** 
  Combines **FinBERT** (a transformer model fine-tuned on financial text for high-precision sentiment polarity) with **VADER** to handle internet slang, emojis, and nuanced financial terminology (bullish, bearish, HODL).
* **Quantitative Market Alignment:** 
  Seamlessly aligns unstructured daily social media sentiment indices with structured historical market return percentages via `yfinance` to uncover potential behavioral finance patterns.

---

## 📂 Project Architecture

```text
reddit-financial-sentiment/
│
├── src/
│   ├── __init__.py
│   ├── scraper.py          # Lightweight RSS-based Reddit post collector
│   ├── sentiment_engine.py # Hybrid FinBERT & VADER sentiment scoring logic
│   ├── market_data.py      # Historical stock price retrieval & return computation (yfinance)
│   └── analysis.py         # Statistical correlation metrics & visualization
│
├── main.py                 # End-to-end pipeline orchestration script
├── requirements.txt        # Project dependencies
├── .gitignore              # Excludes cache, logs, and sensitive configurations
└── README.md               # Project documentation

```

---

## ⚙️ Installation & Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/chakming6767-lgtm/reddit-financial-sentiment.git](https://github.com/chakming6767-lgtm/reddit-financial-sentiment.git)
cd reddit-financial-sentiment

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Run the pipeline:
```bash
python main.py

```



---

## 📊 Methodology & Pipeline Workflow

1. **Data Collection (`scraper.py`):** Pulls live unstructured discussions from targeted subreddits efficiently utilizing public RSS endpoints.
2. **Sentiment Scoring (`sentiment_engine.py`):** Feeds textual data into the FinBERT pipeline and rule-based VADER engine to extract robust compound polarity scores (Positive, Negative, Neutral).
3. **Market Alignment (`market_data.py`):** Fetches matching daily historical price actions and computes percentage returns.
4. **Statistical Analysis (`analysis.py`):** Computes correlation coefficients between daily aggregated sentiment metrics and market returns.

---

## 🎯 Future Roadmap & Scalability

* [ ] **Data Persistence:** Introduce SQLite database integration to accumulate long-term historical time-series data.
* [ ] **Engagement Weighting:** Incorporate post score/comment volume weighting into the daily sentiment index calculation.
* [ ] **Lagged Correlation Analysis:** Evaluate predictive power using time-lagged statistical models (Lead-Lag relationship).

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
