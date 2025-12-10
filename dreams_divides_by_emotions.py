import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path


def analyze_dream_sentiment():
    # --- 1. טעינת הנתונים ---
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parents[2]

    # ננסה לטעון את הקובץ שיצרנו בשלב הקודם (כדי לשלב את הנושאים עם הרגשות)
    possible_files = [
        project_root / "dreams_auto_categorized.csv",  # העדיפות הראשונה - הקובץ עם הקטגוריות
        project_root / "all_dreams_combined.csv"
    ]

    data_file = next((f for f in possible_files if f.exists()), None)

    if not data_file:
        print("Error: Could not find data file. Run the previous scripts first.")
        return

    print(f"Loading data from {data_file.name}...")
    df = pd.read_csv(data_file)

    # מציאת עמודת טקסט
    text_col = next((col for col in ['report', 'content', 'dream', 'description'] if col in df.columns), None)
    if not text_col:
        text_col = max(df.select_dtypes(include=['object']), key=lambda c: df[c].astype(str).str.len().mean())

    df = df.dropna(subset=[text_col])

    # --- 2. אתחול המודל (VADER) ---
    print("Downloading sentiment lexicon...")
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    analyzer = SentimentIntensityAnalyzer()

    print(f"Analyzing emotions in {len(df)} dreams...")

    # --- 3. חישוב הציון לכל חלום ---
    # הפונקציה מחזירה ציון בין -1 (הכי שלילי) לבין +1 (הכי חיובי)
    # 0 זה ניטרלי
    df['sentiment_score'] = df[text_col].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])

    # תרגום הציון למילים
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
        print(f"{label:<25}: {count} dreams ({count / total:.1%})")

    # --- 5. הצגת דוגמאות קיצוניות ---
    print("\n" + "=" * 50)
    print("       🏆 EXTREME DREAMS FOUND")
    print("=" * 50)

    # החלום הכי חיובי
    happiest_idx = df['sentiment_score'].idxmax()
    happiest_dream = df.loc[happiest_idx, text_col]
    print(f"\n😊 THE HAPPIEST DREAM (Score: {df.loc[happiest_idx, 'sentiment_score']}):")
    print(f"\"{happiest_dream[:300]}...\"")

    # החלום הכי שלילי
    scariest_idx = df['sentiment_score'].idxmin()
    scariest_dream = df.loc[scariest_idx, text_col]
    print(f"\n😱 THE SCARIEST DREAM (Score: {df.loc[scariest_idx, 'sentiment_score']}):")
    print(f"\"{scariest_dream[:300]}...\"")

    # --- 6. שמירה ---
    output_path = project_root / "dreams_with_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved full analysis to: {output_path}")


if __name__ == "__main__":
    analyze_dream_sentiment()