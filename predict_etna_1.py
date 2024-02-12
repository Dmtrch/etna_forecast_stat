import pandas as pd
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import FourierTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import MeanTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import TimeSeriesImputerTransform
from etna.transforms import TrendTransform
from etna.datasets import TSDataset
from etna.analysis import plot_backtest
from read_data_statistic import get_df_corrected_filtered_transposed
from etna.analysis import plot_forecast

# Подготовка данных в формате, совместимом с ETNA
# Предполагается, что df_corrected_filtered_transposed уже загружен и содержит необходимые временные ряды

# Создание экземпляра TSDataset для целевого ряда и экзогенных переменных
df_corr_fil = get_df_corrected_filtered_transposed()



# После транспонирования и всех предыдущих преобразований, где 'timestamp' является индексом
#df_corrected_filtered_transposed.reset_index(inplace=True)  # Сброс индекса, чтобы 'timestamp' стал столбцом

df_long = pd.melt(df_corr_fil.reset_index(), id_vars=['timestamp'],
                  var_name='segment', value_name='target')

df_long_ts_format = TSDataset.to_dataset(df_long)
# Создание TSDataset

ts = TSDataset(df=df_long_ts_format,freq='MS')

# Choose a horizon
HORIZON = 14

# Make train/test split
train_ts, test_ts = ts.train_test_split(test_size=HORIZON)

# print(df_long.head())
# print(df_long.dtypes)
#
# print(df_long_ts_format.head())
# print(df_long_ts_format.dtypes)
print(ts.head())


# # Определение трансформаций
# transforms = [
#     LagTransform(in_column='target', lags=[1, 3, 6]),
#     DateFlagsTransform()
# ]
transforms = [
    DensityOutliersTransform(in_column="target", distance_coef=3.0),
    TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
    LinearTrendTransform(in_column="target"),
    TrendTransform(in_column="target", out_column="trend"),
    LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
    DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
    FourierTransform(period=12, order=6, out_column="fourier"),
    SegmentEncoderTransform(),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=12, seasonality=7),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=7),
]


# Создание модели линейной регрессии для каждого сегмента
#model = LinearPerSegmentModel()
model = CatBoostMultiSegmentModel()

# Создание и обучение пайплайна
# pipeline = Pipeline(model=model, transforms=transforms)
# pipeline.fit(ts)

pipeline = Pipeline(model=model, transforms=transforms, horizon=HORIZON)
pipeline.fit(train_ts)

# Make a forecast
forecast_ts = pipeline.forecast()


plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=50)


# # Прогнозирование
# future_ts = ts.make_future(12)  # Прогноз на 12 месяцев вперёд
# forecast = pipeline.forecast(future_ts)
#
# # Визуализация результатов
# plot_backtest(ts, forecast, model=model)
