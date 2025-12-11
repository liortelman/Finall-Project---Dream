import pandas as pd
import numpy as np
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

# --- Custom Dream Dictionaries ---
# These words specifically trigger "Nightmare" or "Bliss" logic
# VADER might miss the context of "falling" or "chased", so we boost them here.
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
    """
    Combines VADER sentiment with custom Dream Keyword counting.
    Returns a score between -1.0 (Scary) and 1.0 (Happy).
    """
    if not isinstance(text, str):
        return 0.0

    # 1. Base VADER Score
    vader_score = analyzer.polarity_scores(text)['compound']

    # 2. Keyword Analysis
    text_lower = text.lower()

    # Count occurrences (simple heuristic)
    nightmare_hits = sum(1 for word in NIGHTMARE_KEYWORDS if word in text_lower)
    bliss_hits = sum(1 for word in BLISS_KEYWORDS if word in text_lower)

    # 3. Adjust Score
    # If we find nightmare words, we drag the score down.
    # If we find bliss words, we push it up.
    # The multiplier (0.15) determines how much impact keywords have.
    adjustment = (bliss_hits * 0.15) - (nightmare_hits * 0.15)

    final_score = vader_score + adjustment

    # Clip results to stay within -1 to 1 range
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
        "dreams_auto_categorized.csv",
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

    # Apply the custom function
    df['sentiment_score'] = df[text_col].apply(lambda x: calculate_hybrid_score(str(x), analyzer))
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)

    # --- 4. Statistics ---
    print("\n" + "=" * 60)
    print("       📊 IMPROVED EMOTIONAL ANALYSIS")
    print("=" * 60)

    # Sort categories in a logical order for printing
    order = ["🌟 Blissful / Happy", "🙂 Positive", "😐 Neutral", "😟 Negative", "💀 Nightmare / Scary"]
    counts = df['sentiment_label'].value_counts()
    total = len(df)

    for label in order:
        count = counts.get(label, 0)
        percent = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(percent / 5)
        print(f"{label:<20} : {count:>4} ({percent:>5.1f}%) {bar}")

    # --- 5. Extract Extremes ---
    print("\n" + "=" * 60)
    print("       🏆 EXTREME DREAM EXAMPLES")
    print("=" * 60)

    # Happiest
    happiest_idx = df['sentiment_score'].idxmax()
    print(f"\n🌟 THE HAPPIEST DREAM (Score: {df.loc[happiest_idx, 'sentiment_score']:.2f}):")
    print(f"\"{str(df.loc[happiest_idx, text_col])[:350]}...\"")

    # Scariest
    scariest_idx = df['sentiment_score'].idxmin()
    print(f"\n💀 THE SCARIEST DREAM (Score: {df.loc[scariest_idx, 'sentiment_score']:.2f}):")
    print(f"\"{str(df.loc[scariest_idx, text_col])[:350]}...\"")

    # Random Neutral (to verify validity)
    try:
        neutral_sample = df[df['sentiment_label'] == "😐 Neutral"].sample(1)
        if not neutral_sample.empty:
            print(f"\n😐 SAMPLE NEUTRAL DREAM:")
            print(f"\"{str(neutral_sample.iloc[0][text_col])[:200]}...\"")
    except:
        pass

    # --- 6. Save ---
    output_path = current_dir / "dreams_sentiment_enhanced.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Analysis saved to: {output_path}")


if __name__ == "__main__":
    analyze_dream_sentiment_improved()