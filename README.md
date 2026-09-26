# Machine Learning Journey

Репозиторий для структурированного изучения машинного обучения. Здесь собираются реализации алгоритмов с нуля (на чистом `NumPy`).

---

## Структура репозитория

```text
machine-learning-journey/
│
├── splitters/              # Собственный класс для безопасного разбиения данных
│   └── splitter.py         # (Train / Validation / Test с использованием np.random.default_rng)
│
├── Linear_model/           # Реализации линейных моделей на NumPy
│   ├── linear_regression.py # Пакетный и стохастический градиентный спуск (SGD)
│   └── logistic_regression.py # Логистическая регрессия с сигмоидой и Log Loss
