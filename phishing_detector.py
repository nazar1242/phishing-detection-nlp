import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

file_path = 'Enron_mini.csv'
try:
    df = pd.read_csv(file_path)
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['full_text'] = df['subject'] + " " + df['body']
    print(f"Файл успішно знайдено за адресою: {file_path}")
except Exception as e:
    print(f"Помилка: Не вдалося знайти {file_path} ({e})")
    exit()

# (NLP)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['full_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n" + "="*40)
print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ:")
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ІНТЕРАКТИВНА ПЕРЕВІРКА
print("\n" + "="*40)
print("ПЕРЕВІРКА МОДЕЛІ!")
while True:
    user_input = input("\nВведи текст листа (або 'exit' для виходу): ")
    if user_input.lower() == 'exit':
        break
    
    vec_input = vectorizer.transform([user_input])
    prob = model.predict_proba(vec_input)[0][1] # Ймовірність фішингу
    pred = model.predict(vec_input)[0]
    
    status = " ФІШИНГ" if pred == 1 else " ЛЕГІТИМНИЙ"
    print(f"Результат аналізу: {status} (Ймовірність атаки: {prob*100:.2f}%)")
