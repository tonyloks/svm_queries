import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Открываем данные
df = pd.read_excel('keyFile.xlsx')

# 2. Нормализация текста
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['Запрос'] = df['Запрос'].apply(normalize_text)

# 3. Векторизация текста
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['Запрос'])

# 4. Создание тестовой модели support vector classifier
y = df['encoded_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)


c_levels = [10]

for C in c_levels:
    clf = SVC(kernel='rbf', probability=True,C=C)
    # 5. Обучение модели
    clf.fit(X_train, y_train)

    # 6. Вывод метрик точности модели
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Результат для уровня C:{C}')
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

#
# Загрузка новых данных для предсказания
new_data = pd.read_excel('new_queries.xlsx')

# Нормализация текста из столбца "Phrase"
new_data['Phrase'] = new_data['Phrase'].apply(normalize_text)

# Векторизация текста
X_new = vectorizer.transform(new_data['Phrase'])

# Предсказание меток с использованием обученной модели
predicted_labels = clf.predict(X_new)

# Добавление предсказанных меток в новый столбец "Predicted_label"
new_data['Predicted_label'] = predicted_labels

# Сохранение результатов в новый файл
new_data.to_excel('new_queries_with_predictions.xlsx', index=False)

print("Predictions added to the new_queries_with_predictions.xlsx file.")
