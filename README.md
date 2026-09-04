# Phishing Email Detection using NLP

A Machine Learning pipeline designed to classify emails as phishing or legitimate using Natural Language Processing (NLP) techniques. This project was developed as a practical implementation of cybersecurity threat detection.

### Model Overview

The model uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization and a **Logistic Regression** classifier. It analyzes linguistic markers, suspicious URLs, and social engineering patterns to detect malicious intent.

- **Dataset:** Enron Email Dataset (Balanced mini-subset of phishing and legitimate emails).
- **Algorithm:** Logistic Regression with `scikit-learn`.
- **Text Processing:** TF-IDF Vectorizer (max 5000 features, English stop-words removed).

### Performance Metrics

The model was evaluated on a held-out 20% test split (400 emails), with the TF-IDF vectorizer fit exclusively on the training data to prevent data leakage.

| Class | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Legitimate (0) | 1.00 | 0.97 | 0.98 | 218 |
| Phishing (1) | 0.96 | 0.99 | 0.98 | 182 |
| **Accuracy** | | | **0.98** | 400 |

**Confusion Matrix:**

| | Predicted Legitimate | Predicted Phishing |
| :--- | :---: | :---: |
| **Actual Legitimate** | 211 | 7 |
| **Actual Phishing** | 1 | 181 |

The model correctly identifies 99% of phishing emails (181/182) while maintaining a low false-positive rate of 3.2% (7/218) on legitimate emails.

### Tech Stack

- `Python 3.x`
- `scikit-learn`
- `pandas`
- `numpy`

### Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/nazar1242/phishing-detection-nlp.git
cd phishing-detection-nlp
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the detector:**
```bash
python phishing_detector.py
```

### Interactive Testing

Upon running the script, the model trains itself and enters an interactive CLI mode. You can paste any email content directly into the terminal to get a real-time probability score of it being a phishing attempt.
