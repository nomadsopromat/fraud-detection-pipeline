import pandas as pd
from scipy.stats import ks_2samp

# "Эталонные" данные — то, на чём обучалась модель
reference_df = pd.read_csv('creditcard.csv')  # или тот CSV, что использовался для train

# "Новые" данные — из продакшн-лога
current_df = pd.read_csv('logs/predictions.csv')

feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']

print(f"{'Feature':<12} {'KS statistic':<15} {'p-value':<12} {'Drift?'}")
print("-" * 55)

drifted_features = []

for col in feature_cols:
    stat, p_value = ks_2samp(reference_df[col], current_df[col])
    drift_detected = p_value < 0.05
    if drift_detected:
        drifted_features.append(col)
    print(f"{col:<12} {stat:<15.4f} {p_value:<12.4f} {'YES' if drift_detected else 'no'}")

print(f"\nВсего признаков с дрейфом: {len(drifted_features)} из {len(feature_cols)}")
if drifted_features:
    print(f"Признаки: {', '.join(drifted_features)}")