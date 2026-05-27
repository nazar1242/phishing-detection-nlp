# 🎣 Phishing Email Detection using NLP

A Machine Learning pipeline designed to classify emails as phishing or legitimate using Natural Language Processing (NLP) techniques. This project was developed as a practical implementation of cybersecurity threat detection.

### 🧠 Model Overview
The model uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization and a **Logistic Regression** classifier. It analyzes linguistic markers, suspicious URLs, and social engineering patterns to detect malicious intent.

- **Dataset:** Enron Email Dataset (Balanced mini-subset of phishing and legitimate emails).
- **Algorithm:** Logistic Regression with `scikit-learn`.
- **Text Processing:** TF-IDF Vectorizer (max 5000 features, English stop-words removed).

### 📊 Performance Metrics
The model was tested on an independent 20% validation split and achieved exceptional results in threat detection:
- **Recall: 0.99** *(The model successfully catches 99% of all malicious emails, minimizing false negatives).*
- High overall accuracy in distinguishing between safe and harmful content.

### 🛠 Tech Stack
- `Python 3.x`
- `scikit-learn`
- `pandas`
- `numpy`

### 🚀 Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/nazar1242/phishing-detection-nlp.git](https://github.com/nazar1242/phishing-detection-nlp.git)
cd phishing-detection-nlp
