# Misinformation Detection with NLP

A practical NLP project to **classify news/posts as TRUE or FAKE** using both classic ML (TF-IDF + classifiers) and a fine-tuned **DistilBERT** transformer. The notebook includes data preprocessing, EDA, multiple modeling approaches, and thorough evaluation.

## Project Overview

- **Goal:** Binary text classification (`TRUE` vs `FAKE`) for misinformation detection.  
- **Approach:**  
  1) Traditional pipeline with **TF-IDF** features + **Logistic Regression / SVM / Random Forest**  
  2) **Transformer fine-tuning** with **DistilBERT** (Hugging Face `Trainer`)  
- **Evaluation:** `accuracy`, `precision/recall/F1` (classification report), **confusion matrix**, **ROC & AUC**.

## Dataset

Notebook expects two CSV files and merges them with a label:

```
/content/true/DataSet_Misinfo_TRUE.csv   # labeled TRUE
/content/fake/DataSet_Misinfo_FAKE.csv   # labeled FAKE
```

> You can change these paths in Section **1. Data** of the notebook.  
> Columns expected: one text column (news/content) and a label constructed as `TRUE` / `FAKE` (the notebook adds the label when reading).

## Methodology

1. **Data Preprocessing**
   - Load & merge TRUE/FAKE CSVs
   - Basic cleaning (lowercasing, punctuation/whitespace handling)
   - Train/validation/test split

2. **Exploratory Data Analysis (EDA)**
   - Class balance, length distributions
   - Word/character frequency, quick text stats
   - Correlation-style visuals for classical features

3. **Modeling**
   - **Classic ML (TF-IDF):** Logistic Regression, SVM, Random Forest  
     - Optional grid search on key hyperparameters
   - **Transformer (DistilBERT):**
     - Tokenization (`DistilBertTokenizer`)
     - Fine-tuning with `Trainer` on the merged dataset

4. **Evaluation & Visualization**
   - `classification_report` (per-class metrics)
   - **Confusion Matrix**
   - **ROC Curve** & **AUC**

## Key Findings (from the notebook)

- Transformer (**DistilBERT**) typically **outperforms** classical baselines on validation/test splits.  
- TF-IDF + Logistic Regression/SVM provide **strong baselines** and are fast to train.  
- Clear separability exists between TRUE/FAKE after preprocessing; ROC AUC highlights the gap.

(*See final cells in the notebook for the exact numbers and plots.*)

## Tech Stack

- **Python**, **PyTorch**, **Transformers (Hugging Face)**, **datasets**
- **scikit-learn**, **pandas**, **numpy**
- **matplotlib**, **seaborn**
- (Utilities) `tqdm`, `psutil`, `gc`

## Repository Files

- `Misinformation_Detection_with_NLP.ipynb` — main notebook: data → EDA → models → evaluation  
- `data/` *(optional structure if you run locally)*  
  - `true/DataSet_Misinfo_TRUE.csv`  
  - `fake/DataSet_Misinfo_FAKE.csv`  
- `figures/` *(optional, if you save plots)*

## How to Run

### Option A — Google Colab (recommended, GPU available)
1. Open the notebook in Colab.  
2. Upload CSVs to the matching paths (or mount Drive and update paths in **Section 1. Data**).  
3. (Optional) If fine-tuning DistilBERT with private models, run:
   ```python
   from huggingface_hub import login
   login()  # paste your HF token securely when prompted
   ```
4. Run all cells top-to-bottom. Plots and metrics will be displayed inline.

### Option B — Local (VS Code / Jupyter)
1. Create environment and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
   ```
2. Place data under:
   ```
   data/true/DataSet_Misinfo_TRUE.csv
   data/fake/DataSet_Misinfo_FAKE.csv
   ```
   and update the notebook paths from `/content/...` → `data/...`.
3. Launch Jupyter:
   ```bash
   jupyter lab   # or jupyter notebook
   ```
4. Run the notebook. For GPU training, ensure CUDA-capable PyTorch is installed.

## Repro Tips

- **Determinism:** set random seeds for NumPy/PyTorch/transformers where provided.  
- **Token safety:** never hardcode tokens in notebooks; use `login()` or env vars.  
- **Speed:** clear outputs before committing; prefer saving heavy figures to `figures/`.

## Results & Reports

- Metrics and plots (confusion matrix, ROC) are produced at the end of each modeling section.  
- Compare classical vs transformer performance to choose a production candidate.

## License

This repository is for educational purposes. If you plan to use the code or data beyond coursework/research, review dataset licenses and adapt a suitable project license.

# Additional Information

For any questions or issues, please open an issue in the repository or contact us at [Kiet-Truong](mailto:truonghongkietcute@gmail.com).

Feel free to customize the project names, descriptions, and any other details specific to your projects. If you encounter any problems or have suggestions for improvements, don't hesitate to reach out. Your feedback and contributions are welcome!

Let me know if there’s anything else you need or if you have any other questions. I’m here to help!
