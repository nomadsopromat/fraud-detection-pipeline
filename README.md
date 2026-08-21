# Fraud Detection Pipeline

End-to-end пайплайн для детекции мошеннических транзакций: от EDA и обучения модели до продакшн-сервиса с мониторингом дрейфа данных.

**Датасет:** [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284 807 транзакций, 0.17% мошеннических.

## Результаты

| Модель | PR-AUC | Precision | Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 0.76 | 0.07 | 0.89 |
| **XGBoost (class weights)** | **0.80** | **0.84** | **0.76** |
| XGBoost + SMOTE | 0.78 | 0.50 | 0.77 |

Финальная модель — XGBoost с `scale_pos_weight`, threshold = 0.3 (подобран по precision-recall кривой).

## Стек

- **Модель:** XGBoost, scikit-learn
- **Интерпретация:** SHAP (глобальная и локальная важность признаков)
- **Сервис:** FastAPI + Docker
- **Мониторинг:** логирование предсказаний, детекция data drift (KS-test)

## Структура проекта

```
├── main.py              # FastAPI сервис
├── Dockerfile
├── requirements.txt
├── models/               # обученная модель + scaler'ы
├── check_drift.py        # проверка дрейфа данных
├── simulate_traffic.py   # симуляция продакшн-трафика
└── get_demo_samples.py   # генерация примеров для демо
```

## Запуск

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs fraud-detection-api
```

API доступен на `http://127.0.0.1:8000/docs` (Swagger UI).

## Что не вошло в репозиторий

`creditcard.csv` и `logs/` — не в репозитории (см. `.gitignore`), скачиваются/генерируются отдельно.