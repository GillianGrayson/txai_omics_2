# Построение моделей и объяснение принятия решений для эпигенетических, когнитивных и иммунологических данных

## Общие сведения

Репозиторий с исходным кодом для Экспериментального Образца (ЭО) программного обеспечения, разрабатываемого в рамках второго этапа научно-исследовательской работы по теме «Приложение методов доверенного и объяснимого искусственного интеллекта к анализу омикс-данных».

## Подготовка окружения

```bash
# clone project
cd <project directory>

# [OPTIONAL] create conda environment
conda create -n env_name python=3.9
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Описание данных и экспериментов
Исследуются данные трех типов: эпигенетические, когнитивные, иммунологические данные.

### Эпигенетические данные
Данные метилирования ДНК цельной крови человека из репозитория [GEO](https://www.ncbi.nlm.nih.gov/geo/) со следующими условиями: крупнейшие наборы данных, которые включают как пациентов с болезнью Паркинсона или шизофренией, так и контрольную группу, при этом в каждой группе не менее 50 семплов.

Для болезни Паркинсона наборы данных GSE145361 и GSE111629 были выбраны в качестве тренировочных, GSE72774 - в качестве тестового.
Всего 2968 семплов (2460 в тренировочной выборке и 508 в тестовой).
Всего было выделено 50911 признаков (CpG-сайтов), которые имеют статистически похожие распределения в тренировочных наборах данных.
Эксперимент заключается в решении задачи классификации Cases (болезнь Паркинсона) vs Controls (здоровые) по данным отобранных признаков (CpG-сайтов).

Для шизофрении наборы данных GSE84727 и GSE 80417 были выбраны в качестве тренировочных, GSE152027 и GSE116379 в качестве тестовых.
Всего 2120 семплов (1522 в тренировочной выборке и 598 в тестовой).
Всего было выделено 1101137 признаков (CpG-сайтов), которые имеют статистически похожие распределения в тренировочных наборах данных.
Эксперимент заключается в решении задачи классификации Cases (шизофрения) vs Controls (здоровые) по данным отобранных признаков (CpG-сайтов).

### Когнитивные данные

Оригинальные данные представляют собой 10-секундные сигналы ЭЭГ, записанные с 32 электродов (триалы).
Для каждого человека для каждого типа движения записывается порядка 20 триалов.
Семплами в задачах машинного обучения являются триалы.
Признаками являются частотные и спектральные характеристики сигналов в общем количестве 320 штук.
В задаче классификации реальных движений левой и правой руки тренировочными данными являются 464 триала, полученные для 12 субъектов, а тестовыми - 105 триалов для 3 независимых субъектов.
В задаче классификации квази движений левой и правой руки тренировочными данными являются 459 триалов, полученные для 12 субъектов, а тестовыми - 115 триалов для 3 независимых субъектов.

### Иммунологические данные
Данные иммунологического профиля представляют из себя значения концентраций 46 цитокинов в плазме крови для 260 человек.
В контексте этих данных решается задача регрессии хронологического возраста по данным иммунологии.
В эксперименте реализуется повторяющаяся кросс-валидация с 5 сплитами и 5 повторами.

## Файловая структура
Директория `data` содержит данные согласно иерархии:
```
└── data                            <- Общая директория с данными
    ├── dnam                           <- Эпигенетические данные
    │   ├── parkinson_classification      <- Эксперимент по классификации Parkinson vs Здоровые
    │   │   ├── models                       <- Результаты экспериментов для разных моделей
    │   │   ├── data.xlsx                    <- Датасет
    │   │   ├── feats_con_*.xlsx             <- Файл с указанием входных признаков
    │   │   └── classes.xlsx                 <- Файл с указанием классов
    │   └── schizophrenia_classification   <- Эксперимент по классификации Schizophrenia vs Здоровые
    │   │   ├── models                        <- Результаты экспериментов для разных моделей
    │   │   ├── data.xlsx                     <- Датасет
    │   │   ├── feats_con_*.xlsx              <- Файл с указанием входных признаков
    │   │   └── classes.xlsx                  <- Файл с указанием классов
    ├── cogn                            <- Когнитивные данные
    │   ├── real_classification            <- Эксперимент по классификации реальных движений
    │   │   ├── models                        <- Результаты экспериментов для разных моделей
    │   │   ├── data.xlsx                     <- Датасет
    │   │   ├── feats_con_*.xlsx              <- Файл с указанием входных признаков
    │   │   └── classes.xlsx                  <- Файл с указанием классов
    │   └── quasi_classification           <- Эксперимент по классификации квази движений
    │   │   ├── models                        <- Результаты экспериментов для разных моделей
    │   │   ├── data.xlsx                     <- Датасет
    │   │   ├── feats_con_*.xlsx              <- Файл с указанием входных признаков
    │   │   └── classes.xlsx                  <- Файл с указанием классов
    └── immuno                         <- Иммунологические данные
        └── age_regression                <- Эксперимент по регрессии возраста
            ├── models                       <- Результаты экспериментов для разных моделей
            ├── data.xlsx                    <- Датасет
            └── feats_con_*.xlsx             <- Файл с указанием входных признаков
```
> `data.xlsx` - таблица, каждая строка которой является семплом, а столбец - признаком.

> `feats_con_*.xlsx` - файл с указанием признаков.
> Модификация этого файла изменит набор признаков, которые будут использоваться для построения модели.

> `classes.xlsx` - файл с указанием меток класса (для задачи классификации).
> Изменение этого файла позволяет выбрать подмножество субъектов, которые будут участвовать в построении модели.

## Конфигурация экспериментов

### Расположение конфигурационных файлов

Конфигурационные файлы экспериментов находятся в следующих директориях:
```
└── configs
    └── experiment
        ├── dnam                                 <- Эпигенетические данные
        │   ├── parkinson_classification.yaml       <- Конфигурационный файл для классификации Parkinson vs Здоровые
        │   └── schizophrenia_classification.yaml   <- Конфигурационный файл для классификации Schizophrenia vs Здоровые
        ├── cogn                                 <- Когнитивные данные
        │   ├── real_classification.yaml            <- Конфигурационный файл для классификации реальных движений
        │   └── quasi_classification.yaml           <- Конфигурационный файл для классификации квази движений
        └── immuno                               <- Иммунологические данные
            └── age_regression.yaml                 <- Конфигурационный файл для регрессии возраста
```

### Общая часть
Во всех конфигурационных файлах есть общая часть, задающая параметры кросс-валидации, оптимизационных метрик, числа эпох, получения объяснимости и других параметров, значения которых специфичны для конкретной задачи.

```yaml
# Cross-validation params
cv_is_split: <bool>           # Perform cross-validation?
cv_n_splits: <целое число>    # Number of splits in cross-validation.
cv_n_repeats: <целое число>   # Number of repeats in cross-validation.

# Optimization metrics params
optimized_metric: <метрика>   # All metrics listed in src.tasks.metrics.
optimized_mean: <усреднение>  # Optimize mean result across all cross-validation splits? Options: ["", "cv_mean"].
optimized_part: <выборка>     # Optimized data partition. Options: ["val", "tst"].
direction: <min or max>       # Direction of metrics optimization. Options ["min", "max"].

# Run params
max_epochs: <целое число>    # Maximum number of epochs.
patience: <целое число>      # Number of early stopping epochs.
feature_importance: <метод>  # Feature importance method. Options: [none, shap_deep, shap_kernel, shap_tree, native].

# Info params
debug: <bool>                # Is Debug?
print_config: <bool>         # Print config?
print_model: <bool>          # Print model info?
ignore_warnings: <bool>      # Ignore warnings?
test_after_training: <bool>  # Test after training?

# SHAP values params
is_shap: <bool>                   # Calculate SHAP values?
is_shap_save: <bool>              # Save SHAP values?
shap_explainer: <тип explainer>   # Type of explainer. Options: ["Tree", "Kernel", "Deep"].
shap_bkgrd: <тип фоновых данных>  # Type of background data. Options: ["trn", "all", "tree_path_dependent"].

# Plot params
num_top_features: <целое число>  # Number of most important features to plot
num_examples: <целое число>      # Number of samples to plot some SHAP figures
```

### Параметры моделей

Во всех конфигурационных файлах экспериментов предусмотрен выбор модели:
```yaml
model:
  type: <тип модели>
```

В конфигурационном файле эксперимента предусмотрено указание параметров для всех типов моделей в соответствующих блоках:
```yaml

# Elastic Net params [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html]
elastic_net:
  ...

# Logistic Regression params [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html]
logistic_regression:
  ...

# SVM params [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html]
svm:
  ...

# XGBoost params [https://xgboost.readthedocs.io/en/stable/parameter.html]
xgboost:
  ...

# LightGBM params [https://lightgbm.readthedocs.io/en/latest/Parameters.html]
lightgbm:
  ...

# CatBoost params [https://catboost.ai/en/docs/references/training-parameters/]
catboost:
  ...

# Params for all adapted models from widedeep available here:
# [https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/model_components.html]
widedeep_tab_mlp:
  ...

widedeep_tab_resnet:
  ...

widedeep_tab_net:
  ...

widedeep_tab_transformer:
  ...

widedeep_ft_transformer:
  ...

widedeep_saint:
  ...

widedeep_tab_fastformer:
  ...

widedeep_tab_perceiver:
  ...

# Params for all adapted models from pytorch_tabular available here:
# [https://github.com/manujosephv/pytorch_tabular/tree/main/pytorch_tabular/models]
pytorch_tabular_autoint:
  ...

pytorch_tabular_tabnet:
  ...

pytorch_tabular_node:
  ...

pytorch_tabular_category_embedding:
  ...

pytorch_tabular_ft_transformer:
  ...

pytorch_tabular_tab_transformer:
  ...

# DANet params [https://arxiv.org/abs/2112.02962]
danet:
  ...

# NAM params [https://github.com/AmrMKayid/nam]
nam:
  ...
```

### Параметры гиперпараметрической оптимизации
Конфигурационные файлы для гиперпараметрической оптимизации для каждой модели находятся в следующих директориях:
```
└── configs
    └── hparams_search
        ├── elastic_net.yaml
        ├── logistic_regression.yaml
        ├── svm.yaml
        ├── xgboost.yaml
        ...
        ...
        └── nam.yaml
```

В каждом конфигурационном файле необходимо задать целевую оптимизационную метрику, а также параметры [OptunaSweeper](https://hydra.cc/docs/plugins/optuna_sweeper/), осуществляющего перебор параметров модели, указанных в блоке `hydra.sweeper.params`:
```yaml
optimized_metric: <метрика>   # All metrics listed in src.tasks.metrics.
optimized_mean: <усреднение>  # Optimize mean result across all cross-validation splits? Options: ["", "cv_mean"].
direction: <min or max>       # Direction of metrics optimization. Options ["min", "max"].
hydra:
  sweeper:
    irection: <minimize or maximize> # 'minimize' or 'maximize' the objective.
    ...
    params:
      <модель>.<параметр_1>: <варьируемый диапазон параметра_1>
      ...
      <модель>.<параметр_n>: <варьируемый диапазон параметра_n>
```

## Запуск экспериментов

### Одиночный эксперимент

Для запуска одиночного эксперимента необходимо указать ключ `experiment` и присвоить ему значение относительного пути имени файла эксперимента без расширения:
```bash
python run.py experiment=dnam/parkinson_classification
python run.py experiment=dnam/schizophrenia_classification   
python run.py experiment=cogn/real_classification
python run.py experiment=cogn/quasi_classification
python run.py experiment=immuno/age_regression
```

### Гиперпараметрическая оптимизация
Для запуска гиперпараметрической оптимизации, помимо указания ключа `experiment`, необходимо указать флаг `--multirun` и указать файл гиперпараметрической оптимизации для выбранной модели при помощи ключа `hparams_search`:
```bash
python run.py --multirun experiment=<эксперимент> -hparams_search=<модель>
```

## Результаты
После запуска и окончания вычислений для выбранного эксперимента файлы и графики с результатами сохраняются в директории `models` выбранного эксперимента в соответствии с иерархией:

```
models
└── <модель>                  # Директория для выбранной модели
    ├── runs                      # Директория с результатами единичных экспериментов
    │   ├── YYYY-MM-DD_HH-MM-SS       # Дата и время запуска
    │   │   ├── shap                      # Директория с результатами объяснимости модели (если is_shap == True)
    │   │   └── ...                       # Результирующие файлы, графики, чекпойнты моделей
    │   └── ...
    └── multiruns                 # Директория с результатами гиперпараметрической оптимизации
        ├── YYYY-MM-DD_HH-MM-SS       # Дата и время запуска
        │   ├── 1                          # Порядковый номер комбинации параметров
        │   │   ├── shap                       # Директория с результатами объяснимости модели (если is_shap == True)
        │   │   └── ...                        # Результирующие файлы, графики, чекпойнты моделей
        │   ├── 2                          # Порядковый номер комбинации параметров
        │   │   ├── shap                       # Директория с результатами объяснимости модели (если is_shap == True)
        │   │   └── ...                        # Результирующие файлы, графики, чекпойнты моделей
        │   └── ...
        └── ...
```
