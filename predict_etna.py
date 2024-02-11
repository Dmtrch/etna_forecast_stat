import pandas as pd
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform
from etna.transforms import DateFlagsTransform
from etna.datasets import TSDataset
from etna.analysis import plot_backtest
from read_data_statistic import get_df_corrected_filtered_transposed

# Подготовка данных в формате, совместимом с ETNA
# Предполагается, что df_corrected_filtered_transposed уже загружен и содержит необходимые временные ряды

# Создание экземпляра TSDataset для целевого ряда и экзогенных переменных
df_corrected_filtered_transposed = get_df_corrected_filtered_transposed()


# После транспонирования и всех предыдущих преобразований, где 'timestamp' является индексом
#df_corrected_filtered_transposed.reset_index(inplace=True)  # Сброс индекса, чтобы 'timestamp' стал столбцом

df_long = pd.melt(df_corrected_filtered_transposed.reset_index(), id_vars=['timestamp'],
                  var_name='segment', value_name='value')

<<<<<<< HEAD

print(df_long.head())
print(df_long.dtypes)


# Создание TSDataset

ts = TSDataset(df=df_long,freq='M')


# ts = TSDataset.to_dataset(df_corrected_filtered_transposed)

# Определение трансформаций
transforms = [
    LagTransform(in_column='volume', lags=[1, 3, 6]),
=======
df_long['target'] = df_long['value']
#df_long['feature'] = 0

df_long_ts_format = TSDataset.to_dataset(df_long)
print(df_long.head())
print(df_long.dtypes)

print(df_long_ts_format.head())
print(df_long_ts_format.dtypes)


# Создание TSDataset

ts = TSDataset(df=df_long_ts_format,freq='MS')

ts.plot()

# Определение трансформаций
transforms = [
    LagTransform(in_column='target', lags=[1, 3, 6]),
>>>>>>> a3980dc (Initial commit)
    DateFlagsTransform()
]

# Создание модели линейной регрессии для каждого сегмента
model = LinearPerSegmentModel()

# Создание и обучение пайплайна
pipeline = Pipeline(model=model, transforms=transforms)
<<<<<<< HEAD
pipeline.fit(ts)

# Прогнозирование
future_ts = ts.make_future(12)  # Прогноз на 12 месяцев вперёд
forecast = pipeline.forecast(future_ts)

# Визуализация результатов
plot_backtest(ts, forecast, model=model)
=======
# pipeline.fit(ts)
#
# # Прогнозирование
# future_ts = ts.make_future(12)  # Прогноз на 12 месяцев вперёд
# forecast = pipeline.forecast(future_ts)
#
# # Визуализация результатов
# plot_backtest(ts, forecast, model=model)
>>>>>>> a3980dc (Initial commit)
