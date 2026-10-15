import xml.etree.ElementTree as ET
import pandas as pd
import requests


def fetch_reddit_posts(
    query: str = "NVDA", subreddit: str = "stocks", pages: int = 5
) -> pd.DataFrame:
  """Fetches real historical Reddit posts and filters to recent data (last 60 days)."""
  posts = []
  session = requests.Session()
  session.headers.update({
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
          " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
      ),
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  })

  subreddits = [subreddit, "wallstreetbets", "investing"]
  cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=60)

  for sub in subreddits:
    print(
        f"[+] Fetching real posts for '{query}' from r/{sub} via RSS stream..."
    )
    rss_url = (
        f"https://www.reddit.com/r/{sub}/search.rss?q={query}&sort=new&limit=100"
    )
    try:
      resp = session.get(rss_url, timeout=10)
      if resp.status_code == 200:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
          title_elem = entry.find("atom:title", ns)
          updated_elem = entry.find("atom:updated", ns)
          id_elem = entry.find("atom:id", ns)

          title = title_elem.text if title_elem is not None else ""
          dt_str = updated_elem.text if updated_elem is not None else ""
          dt = (
              pd.to_datetime(dt_str).tz_localize(None)
              if dt_str
              else pd.Timestamp.now()
          )

          # Ignore archived posts older than 60 days
          if dt < cutoff_date:
            continue

          post_id = (
              id_elem.text.split("/")[-1] if id_elem is not None else title[:10]
          )

          posts.append({
              "post_id": post_id,
              "created_utc": dt,
              "date": dt.strftime("%Y-%m-%d"),
              "title": title,
              "combined_text": title,
              "score": 1,
              "num_comments": 0,
              "subreddit": sub,
          })
    except Exception as e:
      print(f"[!] RSS request for r/{sub} failed: {e}")

  df = pd.DataFrame(posts)

  if not df.empty:
    df = df.drop_duplicates(subset=["post_id"])
    unique_days = df["date"].nunique()
    print(
        f"[+] Successfully gathered {len(df)} real posts across {unique_days}"
        " distinct recent calendar days."
    )
  else:
    print(
        "[!] No recent real posts retrieved. Try broadening query parameters."
    )

  return df