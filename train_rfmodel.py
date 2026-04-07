import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

# 1. Загрузка данных
# Предполагается, что файл находится в той же директории, что и скрипт
df = pd.read_csv('Liver_disease_data.csv')

# 2. Подготовка данных
# Выделение матрицы признаков (X) и вектора целевой переменной (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Разделение на обучающую и тестовую выборки (80/20) с фиксацией seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Инициализация и обучение ансамблевой модели
# n_estimators=100 - стандартное количество деревьев, n_jobs=-1 - использование всех ядер процессора
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. Оценка качества модели
# Для ROC AUC требуются вероятности принадлежности к положительному классу
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Расчет метрики
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# 5. Построение и сохранение графика ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve for Liver Disease Prediction')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_auc_curve.png', dpi=300, bbox_inches='tight')

# 6. Сериализация (сохранение) модели
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_model, model_filename)
print(f"Модель успешно сохранена в файл: {model_filename}")