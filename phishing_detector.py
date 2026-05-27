import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

file_path = r'C:\Users\nvlas\OneDrive\Desktop\Lr\Vstup\Enron.csv'

if not os.path.exists(file_path):
    file_path = 'Enron.csv'

try:
    df = pd.read_csv(file_path)
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['full_text'] = df['subject'] + " " + df['body']
    print(f"Файл успішно знайдено за адресою: {file_path}")
except Exception as e:
    print(f"Помилка: Не вдалося знайти Enron.csv({e})")
    exit()

# (NLP)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['full_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n" + "="*40)
print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ:")
y_pred = model.predict(X_test)
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
