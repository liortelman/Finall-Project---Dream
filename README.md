# Final Project — Dream Analysis 🌙

This project explores the psychological and semantic distribution of dreams using Machine Learning. By leveraging state-of-the-art NLP models and dimensionality reduction, we map raw dream narratives into an interpretable 2D space.

## 🎯 Project Goals

The primary objective is to move beyond simple keyword searching and instead analyze the "latent meaning" of human dreams. Key goals include:

* **Full Dataset Processing:** Scaling the analysis from initial testing phases (1,000 dreams) to processing the complete `all_dreams_combined.csv` dataset.
* **Semantic Mapping:** Converting unstructured text into numerical embeddings that represent underlying psychological themes.
* **Psychological Categorization:** Using PCA to identify axes of "Emotional Intensity" and "Cognitive vs. Experiential" processing.
* **Visual Interpretation:** Creating a "Dream Semantic Space" to visually identify clusters of similar dream types.
* **Data Integrity:** Ensuring robust handling of text data to avoid mathematical errors like divide-by-zero warnings during high-dimensional transformations.

## 🚀 Technical Overview

The pipeline converts a collection of dreams into high-dimensional vectors, which are then projected into a 2D "Semantic Space." This allows us to visualize how dreams cluster together based on emotional intensity and cognitive content.

### Tech Stack
* **Language:** Python 3.9+
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Dimensionality Reduction:** Scikit-learn (PCA)
* **Visualization:** Matplotlib
* **Data Handling:** Pandas



## 📂 Project Structure

* `finel_project_mll.py`: Main execution script for embedding and PCA visualization.
* `all_dreams_combined.csv`: The primary dataset of dream descriptions.
* `dream_semantic_space.png`: The generated PCA visualization plot.
* `/taxonomies`: (Java) Structured data formatters and export utilities.

## 📊 Psychological Mapping

The dreams are categorized into four distinct quadrants based on their PCA coordinates:
1.  **Dramatic + Experiential (Crimson):** High emotional intensity and active experiences.
2.  **Everyday + Experiential (Orange):** Mundane but vivid daily life themes.
3.  **Dramatic + Cognitive (RoyalBlue):** Intense themes processed through an analytical lens.
4.  **Everyday + Cognitive (SeaGreen):** Routine thoughts and logical structures.

## ⚙️ Installation & Usage

Install dependencies via terminal:

```bash
pip install pandas numpy matplotlib sentence-transformers scikit-learn

## Related Links
**project subject**
https://docs.google.com/document/d/1-kmiFr5K4lcD5SQp3Zqv1AJRK5wnGm0i2eaI3yH5Bhk/edit?usp=sharing

**dreams dataset**
 https://dreambank.net/

 **Dictionary of Dreams**
 https://www.kaggle.com/datasets/manswad/dictionary-of-dreams?resource=download

**Report**
 https://www.overleaf.com/project/694af8774d1938e26ca36836
