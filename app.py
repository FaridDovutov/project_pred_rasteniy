import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="Пешгӯии Ҳосилнокӣ / Прогноз Урожайности",
    page_icon="🌱",
    layout="centered"
)

# Функция для загрузки модели
@st.cache_resource
def load_model():
    # Загружаем сохраненную модель
    return joblib.load('best_harvest_model.pkl')

try:
    model_data = load_model()
    model = model_data['model']
    features = model_data['features']
except Exception as e:
    st.error("Хатогӣ дар боркунии модел. Боварӣ ҳосил кунед, ки файли 'best_harvest_model.pkl' мавҷуд аст.")
    # Добавляем вывод технической детали ошибки:
    st.exception(e) 
    st.stop()

# Заголовок интерфейса
st.title("🌱 Системаи пешгӯии ҳосилнокии зироатҳо")
st.write("Интерактивное приложение для прогнозирования урожайности на основе климатических параметров.")

st.markdown("---")

# Создание полей ввода динамически на основе признаков (features), которые ожидает модель
st.subheader("Ворид намудани нишондиҳандаҳо / Ввод параметров")

input_data = {}

# Предположим стандартные параметры (адаптируйте под ваши реальные колонки из X.columns)
# Пример автоматической генерации полей ввода:
for col in features:
    # Определяем дефолтные значения и шаги в зависимости от названий колонок
    if 'temp' in col.lower() or 'ҳарорат' in col.lower():
        min_val, max_val, default = 0.0, 50.0, 25.0
        label = f"Ҳарорат / Температура ({col})"
    elif 'humidity' in col.lower() or 'намӣ' in col.lower():
        min_val, max_val, default = 0.0, 100.0, 60.0
        label = f"Намии ҳаво / Влажность ({col})"
    elif 'water' in col.lower() or 'об' in col.lower():
        min_val, max_val, default = 0.0, 1000.0, 200.0
        label = f"Ҳаҷми обёрӣ / Полив ({col})"
    else:
        min_val, max_val, default = 0.0, 10000.0, 10.0
        label = col

    input_data[col] = st.number_input(label, min_value=min_val, max_value=max_val, value=default, step=0.1)

st.markdown("---")

# Кнопка для запуска прогноза
if st.button("Ҳисоб кардани пешгӯӣ / Рассчитать прогноз", type="primary"):
    # Конвертируем введенные данные в DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Сортируем колонки в правильном порядке, как при обучении
    input_df = input_df[features]
    
    # Делаем предсказание
    prediction = model.predict(input_df)[0]
    
    # Выводим результат
    st.success("### Натиҷаи пешгӯӣ / Результат прогноза:")
    
    # Форматируем вывод (например, центнеров с гектара или тонн)
    st.metric(label="Ҳосилнокии куллӣ (баҳогузорӣ) / Ожидаемая урожайность", value=f"{prediction:.2f} ц/га")
    
    st.balloons()

# Подвал
st.markdown("---")
st.caption("Донишгоҳи давлатии Хоруғ ба номи М. Назаршоев | Факултети математика ва технологияи нав")
