import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import re


def find_file(filename):
    """
    פונקציה חכמה למציאת קבצים (עובדת גם אם הקובץ בתיקייה למעלה)
    """
    current_dir = Path(__file__).resolve().parent
    print(f"🕵️ Searching for '{filename}' starting from: {current_dir}")
    for _ in range(4):
        candidate = current_dir / filename
        if candidate.exists():
            print(f"✅ Found file at: {candidate}")
            return candidate
        if current_dir.parent == current_dir:
            break
        current_dir = current_dir.parent
    return None


def analyze_dreams_by_dictionary():
    # --- 1. טעינת קבצים ---
    dreams_file = find_file("../PCA/all_dreams_combined.csv")
    dict_file = find_file("dreams_interpretations.csv")

    if not dreams_file or not dict_file:
        print("❌ Error: One or more files not found.")
        return

    try:
        df_dreams = pd.read_csv(dreams_file)
        df_dict = pd.read_csv(dict_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 2. הכנת נתוני החלומות ---
    # מציאת עמודת הטקסט
    text_col = next((col for col in ['report', 'content', 'dream', 'description'] if col in df_dreams.columns), None)
    if not text_col:
        text_col = max(df_dreams.select_dtypes(include=['object']),
                       key=lambda c: df_dreams[c].astype(str).str.len().mean())

    # מציאת עמודת החולם (עמודה מס' 3 - אינדקס 2)
    if len(df_dreams.columns) >= 3:
        dreamer_col = df_dreams.columns[2]
    else:
        dreamer_col = None

    df_dreams = df_dreams.dropna(subset=[text_col])

    # חישוב כמות מילים לכל חלום (לצורך הסטטיסטיקה)
    df_dreams['word_count'] = df_dreams[text_col].astype(str).apply(lambda x: len(x.split()))

    # --- 3. הכנת המילון (Dictionary Preparation) ---
    symbol_col = "Dream Symbol"
    meaning_col = "Interpretation"

    # ניקוי הסמלים: הסרת ירידות שורה (\n) והמרה לאותיות קטנות
    df_dict['clean_symbol'] = df_dict[symbol_col].astype(str).str.replace('\n', ' ').str.strip().str.lower()

    # === התיקון כאן: הסרת כפילויות ===
    # אם יש סמלים שחוזרים על עצמם, נשמור רק את המופע הראשון
    initial_len = len(df_dict)
    df_dict = df_dict.drop_duplicates(subset=['clean_symbol'])
    if len(df_dict) < initial_len:
        print(f"⚠️ Removed {initial_len - len(df_dict)} duplicate symbols from dictionary.")

    # יצירת מפה: סמל נקי -> המידע המלא עליו
    symbol_to_data = df_dict.set_index('clean_symbol').to_dict('index')

    # רשימת המילים שנחפש (אוצר המילים)
    vocabulary = list(symbol_to_data.keys())

    print(f"📚 Loaded dictionary with {len(vocabulary)} unique symbols...")
    print(f"🧠 Scanning {len(df_dreams)} dreams for matches...")

    # --- 4. המודל: חיפוש הסמלים (Vectorization) ---
    # אנו משתמשים ב-CountVectorizer כדי למצוא את הסמלים ביעילות
    vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b")

    # הפעלת המודל על כל החלומות בבת אחת
    X = vectorizer.fit_transform(df_dreams[text_col].astype(str))

    feature_names = vectorizer.get_feature_names_out()

    # --- 5. בחירת הפירוש הטוב ביותר לכל חלום ---
    assigned_symbols = []
    assigned_interpretations = []

    for i in range(X.shape[0]):
        # קבלת האינדקסים של הסמלים שנמצאו בחלום הנוכחי
        found_indices = X[i].indices

        if len(found_indices) == 0:
            assigned_symbols.append("Uncategorized")
            assigned_interpretations.append(None)
        else:
            # המרת אינדקסים למילים
            found_words = [feature_names[idx] for idx in found_indices]

            # הלוגיקה: נבחר את הסמל הארוך ביותר שנמצא (הכי ספציפי)
            best_match = max(found_words, key=lambda w: len(w.split()))

            # שליפת המידע המקורי
            original_symbol = symbol_to_data[best_match][symbol_col]
            interpretation_text = symbol_to_data[best_match][meaning_col]

            assigned_symbols.append(original_symbol)
            assigned_interpretations.append(interpretation_text)

    # הוספת התוצאות לטבלה
    df_dreams['matched_symbol'] = assigned_symbols
    df_dreams['interpretation'] = assigned_interpretations

    # --- 6. סטטיסטיקה וסיכום ---
    total_dreams = len(df_dreams)

    # יצירת טבלת סיכום לפי סמלים
    stats = df_dreams.groupby('matched_symbol').agg(
        dreams_count=('matched_symbol', 'count'),
        total_words=('word_count', 'sum'),
        avg_words=('word_count', 'mean'),
        dreamers_count=(dreamer_col, 'nunique') if dreamer_col else ('matched_symbol', lambda x: 0)
    ).reset_index()

    # הוספת טקסט הפירוש לטבלת הסיכום
    def get_interpretation_snippet(symbol):
        if symbol == "Uncategorized": return "No match found"
        clean_s = str(symbol).replace('\n', ' ').strip().lower()
        if clean_s in symbol_to_data:
            return symbol_to_data[clean_s][meaning_col]
        return ""

    stats['Interpretation'] = stats['matched_symbol'].apply(get_interpretation_snippet)
    stats['percent'] = (stats['dreams_count'] / total_dreams) * 100

    # מיון לפי כמות החלומות
    stats = stats.sort_values(by='dreams_count', ascending=False)

    # --- 7. הדפסה ושמירה ---
    print("\n" + "=" * 100)
    print("       📖 TOP 20 INTERPRETATIONS FOUND")
    print("=" * 100)

    header = f"{'Symbol':<20} | {'Dreams':<7} | {'%':<6} | {'Interpretation Snippet'}"
    print(header)
    print("-" * len(header))

    for _, row in stats.head(20).iterrows():
        snippet = str(row['Interpretation'])[:60] + "..."
        print(
            f"{str(row['matched_symbol'])[:20]:<20} | {int(row['dreams_count']):<7} | {row['percent']:>5.1f}% | {snippet}")

    print("=" * 100)

    # שמירה 1: החלומות המלאים עם הפירוש
    output_main = dreams_file.parent / "dreams_interpreted_full.csv"
    df_dreams.to_csv(output_main, index=False)

    # שמירה 2: סיכום סטטיסטי
    summary_df = stats[['matched_symbol', 'Interpretation', 'dreams_count', 'dreamers_count', 'total_words']].copy()
    summary_df.columns = ['Symbol', 'Interpretation', 'how many dreams', 'how many dreamers', 'how many words']

    output_summary = dreams_file.parent / "interpretations_summary.csv"
    summary_df.to_csv(output_summary, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"1. Classified dreams saved to: {output_main}")
    print(f"2. Summary saved to: {output_summary}")


if __name__ == "__main__":
    analyze_dreams_by_dictionary()