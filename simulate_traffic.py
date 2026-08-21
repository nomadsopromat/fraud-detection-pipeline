import pandas as pd
import requests
import time

df = pd.read_csv('creditcard.csv')

# Берём случайную выборку — имитация продакшн-трафика
sample = df.sample(n=200, random_state=1)

url = "http://127.0.0.1:8000/predict"

success = 0
for _, row in sample.iterrows():
    payload = row.drop('Class').to_dict()
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            success += 1
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.05)  # небольшая пауза, чтобы не заваливать сервис мгновенно

print(f"Отправлено успешно: {success}/{len(sample)}")