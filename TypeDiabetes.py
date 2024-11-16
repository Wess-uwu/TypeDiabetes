import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo y los encoders
def load_artifacts():
    modelo_XGBoost = joblib.load(os.path.join('model_artifacts', 'modelo_XGBoost.joblib'))
    data_encoders = joblib.load(os.path.join('model_artifacts', 'data_encoders.joblib'))
    target_encoder = joblib.load(os.path.join('model_artifacts', 'target_encoder.joblib'))
    return modelo_XGBoost, data_encoders, target_encoder

# Cargar los artefactos (sin caché)
modelo_XGBoost, data_encoders, target_encoder = load_artifacts()

# Función para procesar la entrada del usuario
def preprocess_input(user_input, data_encoders):
    # Crear un DataFrame con una sola fila
    input_df = pd.DataFrame([user_input])
    
    # Identificar columnas categóricas
    categorical_cols = data_encoders.keys()
    
    # Aplicar los encoders
    for col, encoder in data_encoders.items():
        if isinstance(encoder, LabelEncoder):  # Cambia esto
            input_df[col] = encoder.transform([user_input[col]])
        elif isinstance(encoder, joblib.encoder.OneHotEncoder):
            # Aplicar OneHotEncoder y concatenar
            ohe_cols = encoder.get_feature_names_out([col])
            encoded = encoder.transform(input_df[[col]]).toarray()
            encoded_df = pd.DataFrame(encoded, columns=ohe_cols)
            input_df = pd.concat([input_df.drop(col, axis=1), encoded_df], axis=1)
    
    return input_df

# Título de la aplicación
st.title("Predicción del Tipo de Diabetes")
st.write("Ingrese los siguientes datos para predecir el tipo de diabetes:")

# Inicializar el contador de predicciones en session_state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

# Crear un formulario para la entrada de datos
with st.form(key='input_form'):
    predisposicion_genetica = st.selectbox('Predisposición Genética', ['Positivo', 'Negativo'])
    presencia_autoanticuerpos = st.selectbox('Presencia de Autoanticuerpos', ['Positivo', 'Negativo'])
    historial_familiar = st.selectbox('Historial Familiar', ['Si', 'No'])
    niveles_insulina = st.number_input('Niveles de Insulina', min_value=0, step=1)
    edad = st.number_input('Edad', min_value=0, step=1)
    imc = st.number_input('IMC', min_value=0.0, step=0.1)
    genero = st.selectbox('Género', ['Masculino', 'Femenino'])
    actividad_fisica = st.selectbox('Actividad Física', ['Baja', 'Moderada', 'Alta'])
    habitos_alimenticios = st.selectbox('Hábitos Alimenticios', ['Saludable', 'No saludable'])
    presión_arterial = st.number_input('Presión Arterial', min_value=0, step=1)
    niveles_colesterol = st.number_input('Niveles de Colesterol', min_value=0, step=1)
    niveles_glucosa = st.number_input('Niveles de Glucosa', min_value=0, step=1)
    factores_socioeconomicos = st.selectbox('Factores Socioeconómicos', ['Baja', 'Mediana', 'Alta'])
    estado_tabaquismo = st.selectbox('Estado de Tabaquismo', ['Fumador', 'No fumador'])
    consumo_alcohol = st.selectbox('Consumo de Alcohol', ['Baja', 'Moderada', 'Alta'])
    salud_pancreas = st.number_input('Salud del Páncreas', min_value=0, step=1)
    funcion_pulmonar = st.number_input('Función Pulmonar', min_value=0, step=1)
    prueba_urinaria = st.selectbox('Prueba Urinaria', ['Cetonas presente', 'Proteína presente', 'Normal', 'Glucosa presente'])
    peso_nacer = st.number_input('Peso al Nacer (gramos)', min_value=0, step=1)
    sintomas_inicio_temprano = st.selectbox('Síntomas de Inicio Temprano', ['Si', 'No'])

    submit_button = st.form_submit_button(label='Predecir')

if submit_button:
    # Recopilar la entrada del usuario
    user_input = {
        'predisposicion_genetica': predisposicion_genetica,
        'presencia_autoanticuerpos': presencia_autoanticuerpos,
        'historial_familiar': historial_familiar,
        'niveles_insulina': niveles_insulina,
        'edad': edad,
        'imc': imc,
        'genero': genero,
        'actividad_fisica': actividad_fisica,
        'habitos_alimenticios': habitos_alimenticios,
        'presión_arterial': presión_arterial,
        'niveles_colesterol': niveles_colesterol,
        'niveles_glucosa': niveles_glucosa,
        'factores_socioeconomicos': factores_socioeconomicos,
        'estado_tabaquismo': estado_tabaquismo,
        'consumo_alcohol': consumo_alcohol,
        'salud_pancreas': salud_pancreas,
        'funcion_pulmonar': funcion_pulmonar,
        'prueba_urinaria': prueba_urinaria,
        'peso_nacer': peso_nacer,
        'sintomas_inicio_temprano': sintomas_inicio_temprano
    }

    # Preprocesar la entrada
    input_processed = preprocess_input(user_input, data_encoders)

    # Realizar la predicción
    prediction_encoded = modelo_XGBoost.predict(input_processed)[0]
    prediction = target_encoder.inverse_transform([prediction_encoded])[0]

    # Incrementar el contador de predicciones
    st.session_state.prediction_count += 1

    # Mostrar el resultado de la predicción con el contador
    st.success(f"El tipo de diabetes predicho es: **{prediction}** - Intento {st.session_state.prediction_count}")
#streamlit run D:\BUSINES\mi_aplicacion_streamlit\app.py