
# Дизайн молекул для лечения болезни Альцгеймера

Репозиторий содержит код для дизайна новых молекул, нацеленных на **гликонегсинтазную киназу-3 бета (Glycogen synthase kinase-3 beta, GSK-3β)** для потенциального лечения **болезни Альцгеймера**.

## Структура проекта

```
.
├── agd/
│   ├── filter/                 # Фильтрация и оценка молекул
│   │   ├── molecule_evaluator.py
│   │   └── sa_scorer.py
│   ├── loader/                 # Загрузка и очистка данных ChEMBL
│   │   ├── load_chembl.py
│   │   └── clean_chembl.py
│   ├── predictor/              # Вычисление признаков, модели и предсказание
│   │   ├── feature_extraction.py
│   │   ├── pipeline.py
│   │   ├── predict.py
│   │   └── processing.py
├── notebooks/                  # Jupyter ноутбуки для анализа
├── resources/                  # Ресурсы и вспомогательные файлы
├── test/                       # Юнит-тесты и тестовые данные
├── scripts/                   # Скрипты для REINVENT4
└── README.md
```

### 1. Установка

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных из базы данных ChEMBL

```python
from agd.loader.load_chembl import parse_activities
from agd.loader.clean_chembl import clean_data

parse_activities("CHEMBL262")      # Загрузка
clean_data("CHEMBL262")         # Удаление пропусков и дубликатов
```

### 3. Генерация признаков

```python
from agd.predictor.feature_extraction import get_features_dataset

df = get_features_dataset(
    target_id="CHEMBL262",
    descriptors=["rdkit", "mordred"],
    fingerprints={
        "morgan": {
            "params": {"radius": 2, "fpSize": 1024}
        }
    }
)
```

Доступные дескрипторы и фингерпринты:
- `rdkit`: стандартные дескрипторы RDKit
- `mordred`: дескрипторы Mordred
- `morgan`: фингерпринты Morgan (с параметрами `radius` и `fpSize`)

### 4. Обучение модели

Шаги обучения модели на основе сгенерированных признаков описаны в `notebooks/training_pipeline.ipynb`. В `resources/` расположена обученная модель `catboost_rdkit_morgan.pkl` (RDKit дескрипторы + Morgan фингерпринты), используется для предсказания PiC50 (R2 = 0.7236).


### 5. Предсказание активности молекул

```python
from agd.predictor.predict import predict_rdkit_morgan

smiles = "COc1ccc(CNC(=O)Nc2ncc([N+](=O)[O-])s2)cc1"
predict_rdkit_morgan(smiles=smiles, model_name="catboost")
```

### 6. Генерация новых молекул с помощью [REINVENT](https://github.com/MolecularAI/REINVENT4)
На основе обученной модели `catboost_rdkit_morgan.pkl` и SMILES-строк из `resources/` сгенерированы новые молекулы с использованием генеративной модели REINVENT.
1. Создан `scripts/reinvent_predict.py`, совместимый с REINVENT.
2. В процессе RL-обучения модели REINVENT4 скрипт использовался как `ExternalProcess` для оценки активности сгенерированных молекул.
3. Проведена генерация новых молекул с учётом таргета и обученной модели.

Результаты генерации молекул: https://drive.google.com/drive/folders/17QwYozkSWYgUe1wXoEDbZXLi7SBCenti?usp=drive_link

### 7. Фильтрация сгенерированных молекул

  * **`pIC50` (предсказанная активность) \>= 6.7**: потенциальная эффективность против GSK-3β
  * **`qed` (Quantitative Estimation of Drug-likeness) \> 0.7**: высокая степень лекарственного подобия
  * **`sa` (Synthetic Accessibility) в диапазоне от 2 до 6**: умеренная синтетическая доступность
  * **`bbb` (Blood-Brain Barrier permeability)**: способность проникать через гематоэнцефалический барьер
  * **`tox_free`**: отсутствие токсичности
  * **`lip` (Lipinski's Rule of Five violations) \<= 1**: соответствие правилам Липинского
  * **`carc`**: отсутствие канцерогенности

Эта система фильтрации гарантирует, что только **качественные, потенциально эффективные и безопасные** молекулы будут отобраны для дальнейшего анализа.


## Контакты

* @SireX2106
* @philareth
* @sunraysu
* @matveiss
* @readKavafisListenFredAgain

