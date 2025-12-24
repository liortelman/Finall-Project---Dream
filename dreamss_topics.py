import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path


def find_data_file():
    current_dir = Path(__file__).resolve().parent
    filename = "all_dreams_combined.csv"
    print(f"🕵️ Searching for '{filename}' starting from: {current_dir}")
    for _ in range(4):
        candidate = current_dir / filename
        if candidate.exists():
            return candidate
        if current_dir.parent == current_dir:
            break
        current_dir = current_dir.parent
    return None


def analyze_dreams_complete():
    # --- 1. טעינת הנתונים ---
    data_file = find_data_file()
    if not data_file:
        print("❌ Error: File not found.")
        return

    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- זיהוי עמודות ---

    # מציאת עמודת טקסט החלום (אוטומטי - העמודה עם הטקסט הכי ארוך)
    text_col = next((col for col in ['report', 'content', 'dream', 'description'] if col in df.columns), None)
    if not text_col:
        text_col = max(df.select_dtypes(include=['object']), key=lambda c: df[c].astype(str).str.len().mean())

    # === תיקון: הגדרת עמודת החולם כעמודה מס' 3 (אינדקס 2) ===
    if len(df.columns) >= 3:
        dreamer_col = df.columns[2]
        print(f"ℹ️ Dreamer ID column detected: '{dreamer_col}' (Column #3)")
    else:
        dreamer_col = None
        print("⚠️ Warning: File has fewer than 3 columns. Cannot identify dreamer column.")

    df = df.dropna(subset=[text_col])

    # חישוב מילים לכל חלום
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    total_dreams_analyzed = len(df)
    print(f"Analyzing {total_dreams_analyzed} dreams...")

    # --- 2. מודל וניקוי ---
    custom_stop_words = list(text.ENGLISH_STOP_WORDS)
    custom_stop_words.extend([
        'dream', 'dreamed', 'dreamt', 'woke', 'awakened', 'remember', 'recall', 'say', 'says', 'look',
        'said', 'realized', 'went', 'got', 'did', 'like', 'just', 'know', 'think', 'saw', 'felt',
        'didn', 'thought', 'don', 'asked', 'told', 'started', 'looking', 'going', 'looked',
        'wanted', 'came', 'couldn', 'saying', 'sent','i',
    ])

    tf_vectorizer = CountVectorizer(
        max_df=0.90, min_df=10, stop_words=custom_stop_words,
        max_features=1500, token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    tf = tf_vectorizer.fit_transform(df[text_col].astype(str))

    n_topics = 100
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tf)

    # --- 3. הכנת שמות הקטגוריות ---
    feature_names = tf_vectorizer.get_feature_names_out()

    topic_map = {}  # מילון לשמירה בקובץ המסכם (מילות מפתח מלאות)
    print_map = {}  # מילון להדפסה (3 מילים ראשונות)

    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # 10 מילים הכי חזקות
        top_words = [feature_names[i] for i in top_words_idx]

        t_id = topic_idx + 1

        full_keywords = ", ".join(top_words)
        short_keywords = ", ".join(top_words[:3])

        topic_map[t_id] = full_keywords
        print_map[t_id] = short_keywords

    # שיוך חלומות
    topic_values = lda.transform(tf)
    df['topic_id'] = topic_values.argmax(axis=1) + 1

    # הוספת מילות מפתח קצרות ל-DF הראשי (עבור הקובץ הראשון וההדפסה)
    df['topic_keywords'] = df['topic_id'].map(print_map)

    # --- 4. חישוב סטטיסטיקה מרוכז ---
    stats = df.groupby('topic_id').agg(
        dreams_count=('topic_id', 'count'),
        total_words=('word_count', 'sum'),
        avg_words=('word_count', 'mean'),
        dreamers_count=(dreamer_col, 'nunique') if dreamer_col else ('topic_id', lambda x: 0)
    ).reset_index()

    # הוספת מילות המפתח המלאות לטבלת הסטטיסטיקה (עבור קובץ הסיכום)
    stats['Keywords'] = stats['topic_id'].apply(lambda x: topic_map[x])

    # הוספת מילות מפתח קצרות להדפסה
    stats['short_keywords'] = stats['topic_id'].apply(lambda x: print_map[x])

    # חישוב אחוזים
    stats['percent'] = (stats['dreams_count'] / total_dreams_analyzed) * 100

    # מיון לפי גודל הקטגוריה
    stats = stats.sort_values(by='dreams_count', ascending=False)

    # --- 5. הדפסה למסך ---
    print("\n" + "=" * 95)
    print("       📊 DREAM STATISTICS (Sorted by Frequency)")
    print("=" * 95)

    header = f"{'ID':<3} | {'Keywords (Top 3)':<30} | {'Dreams':<7} | {'%':<6} | {'Total Words':<11} | {'Avg Length'}"
    print(header)
    print("-" * len(header))

    for _, row in stats.iterrows():
        print(
            f"{row['topic_id']:<3} | {row['short_keywords']:<30} | {int(row['dreams_count']):<7} | {row['percent']:>5.1f}% | {int(row['total_words']):<11,} | {row['avg_words']:.0f} w/dream")

    print("=" * 95)

    # --- 6. שמירת קבצים ---

    # קובץ 1: הדאטה המלא עם סיווג
    output_main = data_file.parent / "dreams_auto_categorized.csv"
    df.to_csv(output_main, index=False)

    # קובץ 2: טבלת סיכום קטגוריות
    summary_df = stats[['topic_id', 'Keywords', 'dreams_count', 'dreamers_count', 'total_words']].copy()

    # שינוי שמות העמודות
    summary_df.columns = ['id', 'Keywords', 'how many dreams', 'how many dreamers', 'how many words']

    output_summary = data_file.parent / "topics_summary.csv"
    summary_df.to_csv(output_summary, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"1. Classified dreams saved to: {output_main}")
    print(f"2. Categories summary saved to: {output_summary}")


if __name__ == "__main__":
    analyze_dreams_complete()