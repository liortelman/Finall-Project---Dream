import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path


def analyze_dream_sentiment():
    # --- 1. טעינת הנתונים (התיקון) ---

    # במקום לחפש בתיקיות למעלה, נחפש בתיקייה שבה הסקריפט נמצא כרגע
    current_dir = Path(__file__).resolve().parent
    print(f"📂 Looking for files in: {current_dir}")

    possible_files = [
        "dreams_auto_categorized.csv",  # עדיפות 1
        "dreams_with_categories_final.csv",  # עדיפות 2
        "all_dreams_combined.csv"  # עדיפות 3
    ]

    data_file = None
    for filename in possible_files:
        candidate = current_dir / filename
        if candidate.exists():
            data_file = candidate
            print(f"✅ Found file: {filename}")
            break

    if not data_file:
        print("\n❌ Error: Could not find any CSV file.")
        print(f"Please make sure one of these files is inside: {current_dir}")
        return

    print(f"Loading data...")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # מציאת עמודת טקסט באופן אוטומטי
    text_col = next((col for col in ['report', 'content', 'dream', 'description'] if col in df.columns), None)
    if not text_col:
        # אם לא מצא לפי שם, לוקח את העמודה עם הטקסט הכי ארוך
        text_col = max(df.select_dtypes(include=['object']), key=lambda c: df[c].astype(str).str.len().mean())

    df = df.dropna(subset=[text_col])

    # --- 2. אתחול המודל (VADER) ---
    print("Initializing sentiment analyzer...")
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER dictionary...")
        nltk.download('vader_lexicon')

    analyzer = SentimentIntensityAnalyzer()

    print(f"Analyzing emotions in {len(df)} dreams...")

    # --- 3. חישוב הציון לכל חלום ---
    df['sentiment_score'] = df[text_col].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])

    def categorize_sentiment(score):
        if score >= 0.05: return "Positive (Happy)"
        if score <= -0.05: return "Negative (Nightmare/Sad)"
        return "Neutral"

    df['sentiment_label'] = df['sentiment_score'].apply(categorize_sentiment)

    # --- 4. הצגת סטטיסטיקות ---
    print("\n" + "=" * 50)
    print("       📊 EMOTIONAL ANALYSIS RESULTS")
    print("=" * 50)

    counts = df['sentiment_label'].value_counts()
    total = len(df)
    for label, count in counts.items():
        print(f"🎭 {label:<25}: {count} dreams ({count / total:.1%})")

    # --- 5. הצגת דוגמאות קיצוניות ---
    print("\n" + "=" * 50)
    print("       🏆 EXTREME DREAMS FOUND")
    print("=" * 50)

    # החלום הכי חיובי
    happiest_idx = df['sentiment_score'].idxmax()
    happiest_dream = str(df.loc[happiest_idx, text_col])
    print(f"\n😊 THE HAPPIEST DREAM (Score: {df.loc[happiest_idx, 'sentiment_score']}):")
    print(f"\"{happiest_dream[:300]}...\"")

    # החלום הכי שלילי
    scariest_idx = df['sentiment_score'].idxmin()
    scariest_dream = str(df.loc[scariest_idx, text_col])
    print(f"\n😱 THE SCARIEST DREAM (Score: {df.loc[scariest_idx, 'sentiment_score']}):")
    print(f"\"{scariest_dream[:300]}...\"")

    # --- 6. שמירה ---
    output_path = current_dir / "dreams_with_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved full analysis to: {output_path}")


if __name__ == "__main__":
    analyze_dream_sentiment()