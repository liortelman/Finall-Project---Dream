import pandas as pd
import numpy as np
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

# --- Custom Dream Dictionaries ---
NIGHTMARE_KEYWORDS = {
    'nightmare', 'terrified', 'scared', 'scream', 'screaming', 'blood', 'kill',
    'death', 'dead', 'monster', 'demon', 'ghost', 'chased', 'running away',
    'trapped', 'paralyzed', 'darkness', 'evil', 'attack', 'attacked', 'gun',
    'knife', 'corpse', 'drowning', 'falling', 'teeth', 'spider', 'snake'
}

BLISS_KEYWORDS = {
    'happy', 'joy', 'love', 'loved', 'beautiful', 'amazing', 'wonderful',
    'peace', 'peaceful', 'flying', 'fly', 'lucid', 'control', 'kiss',
    'hug', 'warmth', 'light', 'angel', 'ecstasy', 'perfect', 'paradise'
}


def calculate_hybrid_score(text, analyzer):
    if not isinstance(text, str):
        return 0.0

    # 1. Base VADER Score
    vader_score = analyzer.polarity_scores(text)['compound']

    # 2. Keyword Analysis
    text_lower = text.lower()
    nightmare_hits = sum(1 for word in NIGHTMARE_KEYWORDS if word in text_lower)
    bliss_hits = sum(1 for word in BLISS_KEYWORDS if word in text_lower)

    # 3. Adjust Score
    adjustment = (bliss_hits * 0.15) - (nightmare_hits * 0.15)
    final_score = vader_score + adjustment

    return max(-1.0, min(1.0, final_score))


def get_sentiment_label(score):
    if score >= 0.3: return "🌟 Blissful / Happy"
    if score >= 0.05: return "🙂 Positive"
    if score <= -0.4: return "💀 Nightmare / Scary"
    if score <= -0.05: return "😟 Negative"
    return "😐 Neutral"


def analyze_dream_sentiment_improved():
    # --- 1. Load Data ---
    current_dir = Path(__file__).resolve().parent
    print(f"📂 Looking for files in: {current_dir}")

    possible_files = [
        "dreams_auto_categorized.csv",  # Best file (has Topic IDs)
        "dreams_with_categories_final.csv",
        "all_dreams_combined.csv"
    ]

    data_file = None
    for filename in possible_files:
        candidate = current_dir / filename
        if candidate.exists():
            data_file = candidate
            print(f"✅ Found file: {filename}")
            break

    if not data_file:
        print("❌ Error: Could not find CSV file.")
        return

    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    text_col = next((col for col in ['report', 'content', 'dream', 'description'] if col in df.columns), None)
    if not text_col:
        text_col = max(df.select_dtypes(include=['object']), key=lambda c: df[c].astype(str).str.len().mean())

    df = df.dropna(subset=[text_col])
    print(f"Analyzing {len(df)} dreams...")

    # --- 2. Initialize VADER ---
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER dictionary...")
        nltk.download('vader_lexicon')

    analyzer = SentimentIntensityAnalyzer()

    # --- 3. Apply Hybrid Analysis ---
    print("running hybrid emotional analysis...")
    df['sentiment_score'] = df[text_col].apply(lambda x: calculate_hybrid_score(str(x), analyzer))
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

    # --- 4. General Statistics ---
    print("\n" + "=" * 60)
    print("       📊 EMOTIONAL ANALYSIS OVERVIEW")
    print("=" * 60)

    order = ["🌟 Blissful / Happy", "🙂 Positive", "😐 Neutral", "😟 Negative", "💀 Nightmare / Scary"]
    counts = df['sentiment_label'].value_counts()
    total = len(df)

    for label in order:
        count = counts.get(label, 0)
        percent = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(percent / 5)
        print(f"{label:<20} : {count:>4} ({percent:>5.1f}%) {bar}")

    # --- 5. CATEGORY STATISTICS (NEW SECTION) ---
    # בודקים אם יש בכלל עמודת ID (ייתכן והקובץ המקורי נטען ללא קטגוריות)
    if 'topic_id' in df.columns:
        print("\n" + "=" * 80)
        print("       🧬 EMOTIONAL ANALYSIS BY TOPIC (Sorted by Scariest to Happiest)")
        print("=" * 80)

        # אם יש מילות מפתח, נשתמש בהן לקבוצה, אחרת רק לפי ID
        group_cols = ['topic_id', 'topic_keywords'] if 'topic_keywords' in df.columns else ['topic_id']

        # חישוב אגרגציה לכל נושא
        topic_stats = df.groupby(group_cols).agg(
            total_dreams=('sentiment_score', 'count'),
            avg_sentiment=('sentiment_score', 'mean'),
            nightmare_count=('sentiment_label', lambda x: (x == "💀 Nightmare / Scary").sum()),
            bliss_count=('sentiment_label', lambda x: (x == "🌟 Blissful / Happy").sum())
        ).reset_index()

        # מיון: מהציון הכי נמוך (הכי מפחיד) לציון הכי גבוה
        topic_stats = topic_stats.sort_values(by='avg_sentiment', ascending=True)

        # כותרת לטבלה
        print(f"{'ID':<3} | {'Avg Score':<9} | {'Scary #':<7} | {'Happy #':<7} | {'Topic Keywords'}")
        print("-" * 80)

        for _, row in topic_stats.iterrows():
            t_id = row['topic_id']
            avg = row['avg_sentiment']
            scary = row['nightmare_count']
            happy = row['bliss_count']

            # אם יש מילות מפתח נציג אותן, אחרת מחרוזת ריקה
            keywords = row['topic_keywords'] if 'topic_keywords' in row else ""

            # עיצוב ויזואלי פשוט לציון
            sentiment_icon = "😱" if avg < -0.2 else ("😍" if avg > 0.2 else "😐")

            print(f"{t_id:<3} | {avg:>6.2f} {sentiment_icon} | {scary:<7} | {happy:<7} | {keywords}")

    # --- 6. Extract Extremes ---
    print("\n" + "=" * 60)
    print("       🏆 EXTREME DREAM EXAMPLES")
    print("=" * 60)

    happiest_idx = df['sentiment_score'].idxmax()
    print(f"\n🌟 THE HAPPIEST DREAM (Score: {df.loc[happiest_idx, 'sentiment_score']:.2f}):")
    print(f"\"{str(df.loc[happiest_idx, text_col])[:300]}...\"")

    scariest_idx = df['sentiment_score'].idxmin()
    print(f"\n💀 THE SCARIEST DREAM (Score: {df.loc[scariest_idx, 'sentiment_score']:.2f}):")
    print(f"\"{str(df.loc[scariest_idx, text_col])[:300]}...\"")

    # --- 7. Save ---
    output_path = current_dir / "dreams_sentiment_enhanced.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Analysis saved to: {output_path}")


if __name__ == "__main__":
    analyze_dream_sentiment_improved()