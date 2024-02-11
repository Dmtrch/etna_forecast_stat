import pandas as pd

# Загрузка данных из файла
file_path = 'statistic_14-23.xlsx'
df = pd.read_excel(file_path)

# Фильтрация необходимых категорий
categories = [
    'Курс $',
    'Кокс, Австралия, FOB, $',
    'Чугун, FOB, $',
    'Лом, FOB, $',
    'Заготовка РФ, черное море, FOB, $',  # Предполагаемая категория для металлической квадратной заготовки
    'Арматура, черное море, FOB, $'       # Предполагаемая категория для металлической арматуры
]
df_filtered = df[df['timestamp'].isin(categories)]


# Транспонирование DataFrame, чтобы даты стали индексами, а категории - столбцами
df_transposed = df_filtered.set_index('timestamp').transpose()

# Преобразование индекса в datetime
df_transposed.index = pd.to_datetime(df_transposed.index)

# Присвоение имени 'timestamp' индексу
df_transposed.index.name = 'timestamp'

# Переименование столбцов
df_transposed.columns = [
    'Exchange_Rate_USD',
    'Coking_Coal_Price',
    'Pig_Iron_Price',
    'Scrap_Metal_Price',
    'Square_Billet_Price',
    'Rebar_Price'
]

# Обработка пропущенных значений (для демонстрации заполним пропуски нулями)
df_transposed.fillna(0, inplace=True)

# Вывод первых строк обработанного DataFrame
#print(df_transposed.head())

def split_time_series(target_series, exog_data, split_ratio=0.8):
    """
    Разделяет временной ряд и соответствующие экзогенные данные на обучающий и тестовый наборы.

    Параметры:
    - target_series: pd.Series - целевой временной ряд.
    - exog_data: pd.DataFrame - экзогенные переменные.
    - split_ratio: float - доля данных для обучающего набора.

    Возвращает:
    - train_target: pd.Series - обучающий набор для целевого временного ряда.
    - test_target: pd.Series - тестовый набор для целевого временного ряда.
    - train_exog: pd.DataFrame - обучающий набор для экзогенных переменных.
    - test_exog: pd.DataFrame - тестовый набор для экзогенных переменных.
    """
    split_point = int(len(target_series) * split_ratio)

    train_target = target_series[:split_point]
    test_target = target_series[split_point:]
    train_exog = exog_data[:split_point]
    test_exog = exog_data[split_point:]

    return train_target, test_target, train_exog, test_exog


train_square_billet, test_square_billet, train_exog_square_billet, test_exog_square_billet = split_time_series(
    target_series=df_transposed['Square_Billet_Price'],
    exog_data=df_transposed[['Exchange_Rate_USD', 'Pig_Iron_Price', 'Scrap_Metal_Price']]
)

# Пример использования функции для разделения данных "цены металлической арматуры"
train_rebar, test_rebar, train_exog_rebar, test_exog_rebar = split_time_series(
    target_series=df_transposed['Rebar_Price'],
    exog_data=df_transposed[['Scrap_Metal_Price']]
)

def generate_features(target_series, exog_data=None, lags=[1, 3, 6], rolling_window=3):
    """
    Генерирует лаги и скользящие средние для целевого временного ряда и экзогенных переменных.

    Параметры:
        target_series (pd.Series): Целевой временной ряд.
        exog_data (pd.DataFrame): DataFrame с экзогенными переменными. Может быть None.
        lags (list of int): Список значений лагов для генерации.
        rolling_window (int): Размер окна для скользящего среднего.

    Возвращает:
        pd.DataFrame: DataFrame с сгенерированными признаками.
    """
    # Генерация лагов и скользящего среднего для целевого ряда
    features = pd.DataFrame(index=target_series.index)
    for lag in lags:
        features[f'target_lag_{lag}'] = target_series.shift(lag)
    features['target_rolling_mean'] = target_series.rolling(window=rolling_window).mean()

    # Генерация лагов и скользящего среднего для экзогенных переменных, если они есть
    if exog_data is not None:
        for column in exog_data.columns:
            for lag in lags:
                features[f'{column}_lag_{lag}'] = exog_data[column].shift(lag)
            features[f'{column}_rolling_mean'] = exog_data[column].rolling(window=rolling_window).mean()

    return features.dropna()


# В конце обработки данных, добавьте следующую строку:
df_corrected_filtered_transposed = df_transposed



# Экспорт переменной, если это необходимо для использования в других модулях:
def get_df_corrected_filtered_transposed():
    return df_corrected_filtered_transposed

# Применение функции для "цены квадратной заготовки" с экзогенными переменными
square_billet_features = generate_features(
    target_series=train_square_billet,
    exog_data=train_exog_square_billet
)

# Применение функции для "цены металлической арматуры" с экзогенными переменными
rebar_features = generate_features(
    target_series=train_rebar,
    exog_data=train_exog_rebar
)

# Вывод размеров полученных наборов данных для проверки
#print(f"Квадратная заготовка: Обучение - {len(train_square_billet)}, Тест - {len(test_square_billet)}")
#print(f"Металлическая арматура: Обучение - {len(train_rebar)}, Тест - {len(test_rebar)}")

# Вывод первых нескольких строк сгенерированных признаков для проверки
#print("Признаки для 'цены квадратной заготовки':")
#print(square_billet_features.head())
#print("\nПризнаки для 'цены металлической арматуры':")
#print(rebar_features.head())
