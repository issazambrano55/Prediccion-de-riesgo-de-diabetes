import streamlit as st     # Esto es para hacer la pagina web interactiva
import pandas as pd        # Para manejar tablas y datos (dataframes)
import joblib              # Para cargar modelos y objetos guardados en archivos


# COMENZAMOS CARGANDO EL MODELO Y LOS DATOS


# Cargamos el modelo que predice riesgo de diabetes
model = joblib.load("modelo_diabetes.pkl")

# Cargamos el "scaler", que sirve para normalizar los datos (hacerlos comparables)
scaler = joblib.load("scaler.pkl")

# Cargamos el "umbral óptimo", que es el punto que indica para decir si hay riesgo o no
with open("umbral_optimo.txt", "r") as f:
    umbral_optimo = float(f.read())


# EL LISTADO DE CARACTERÍSTICAS


# Orden de las características que el modelo espera recibir
# Estas son las columnas que el modelo utiliza para hacer la predicción
features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Aquí damos nombres más intuitivos a cada columna para mostrar al usuario
feature_labels = {
    "HighBP": "Hipertensión",
    "HighChol": "Colesterol alto",
    "CholCheck": "Chequeo de colesterol en últimos 5 años",
    "BMI": "Índice de Masa Corporal (BMI)",
    "Smoker": "Fumador",
    "Stroke": "Derrame cerebral",
    "HeartDiseaseorAttack": "Enfermedad cardíaca o ataque",
    "PhysActivity": "Actividad física en últimos 30 días",
    "Fruits": "Consume frutas diariamente",
    "Veggies": "Consume verduras diariamente",
    "HvyAlcoholConsump": "Consumo alto de alcohol",
    "AnyHealthcare": "Tiene acceso a atención médica",
    "NoDocbcCost": "No fue al médico por costo en último año",
    "GenHlth": "Salud general (1=Excelente, 5=Mala)",
    "MentHlth": "Días de mala salud mental (últimos 30 días)",
    "PhysHlth": "Días de mala salud física (últimos 30 días)",
    "DiffWalk": "Dificultad para caminar",
    "Sex": "Sexo",
    "Age": "Edad",
    "Education": "Nivel educativo",
    "Income": "Ingreso anual"
}


# CONFIGURACIÓN DE PÁGINA


# Título y formato de la página
st.set_page_config(page_title="Predicción de Diabetes", layout="wide")
st.title("Evaluación de Riesgo de Diabetes") #Titulo que se muestra en la app web
st.write("Complete el formulario en el siguiente orden para evaluar su riesgo") #Subtitulo de la app web


# FORMULARIO PRINCIPAL


with st.form("diabetes_form"):  # Creamos un formulario con varios campos
    input_data = {}  # Aquí guardaremos las respuestas del usuario
    
 
    # Sección 1: Salud Cardiovascular
   
    st.subheader("Salud Cardiovascular")
    col1, col2 = st.columns(2)  # Dividimos en 2 columnas

    with col1:
        opcion = st.selectbox(feature_labels["HighBP"], ["No", "Sí"])
        input_data["HighBP"] = 1 if opcion == "Sí" else 0

        opcion = st.selectbox(feature_labels["HighChol"], ["No", "Sí"])
        input_data["HighChol"] = 1 if opcion == "Sí" else 0

        opcion = st.selectbox(feature_labels["CholCheck"], ["No", "Sí"])
        input_data["CholCheck"] = 1 if opcion == "Sí" else 0

    with col2:
        opcion = st.selectbox(feature_labels["Stroke"], ["No", "Sí"])
        input_data["Stroke"] = 1 if opcion == "Sí" else 0

        opcion = st.selectbox(feature_labels["HeartDiseaseorAttack"], ["No", "Sí"])
        input_data["HeartDiseaseorAttack"] = 1 if opcion == "Sí" else 0

        opcion = st.selectbox(feature_labels["DiffWalk"], ["No", "Sí"])
        input_data["DiffWalk"] = 1 if opcion == "Sí" else 0

   
    # Sección 2: Medidas Corporales, que sirven para calcular el BMI (Índice de Masa Corporal)
    
    st.subheader("Medidas Corporales")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        peso = st.number_input("Peso (kg)", 30.0, 300.0, 70.0, step=0.5)

    with col2:
        altura = st.number_input("Altura (cm)", 100.0, 250.0, 170.0, step=1.0)

    with col3:
        if altura > 0:
            altura_m = altura / 100
            bmi_calc = round(peso / (altura_m ** 2), 2)  # Fórmula BMI
            st.metric(feature_labels["BMI"], bmi_calc)   # Mostrar BMI
            input_data["BMI"] = bmi_calc


    # Sección 3: Hábitos de Vida, en esta sección se evalúan los hábitos que pueden influir en la salud
   
    st.subheader("Hábitos de Vida")
    col1, col2 = st.columns(2)

    with col1:
        opcion = st.selectbox(feature_labels["Smoker"], ["No", "Sí"])
        input_data["Smoker"] = 1 if opcion == "Sí" else 0 # Convertimos a 1 o 0

        opcion = st.selectbox(feature_labels["PhysActivity"], ["No", "Sí"])
        input_data["PhysActivity"] = 1 if opcion == "Sí" else 0 # Convertimos a 1 o 0

        opcion = st.selectbox(feature_labels["Fruits"], ["No", "Sí"])
        input_data["Fruits"] = 1 if opcion == "Sí" else 0 # Convertimos a 1 o 0

        opcion = st.selectbox(feature_labels["Veggies"], ["No", "Sí"])
        input_data["Veggies"] = 1 if opcion == "Sí" else 0 # Convertimos a 1 o 0

    with col2:
        opcion = st.selectbox(feature_labels["HvyAlcoholConsump"], ["No", "Sí"])
        input_data["HvyAlcoholConsump"] = 1 if opcion == "Sí" else 0  # Convertimos a 1 o 0

        opcion = st.selectbox(feature_labels["AnyHealthcare"], ["No", "Sí"])
        input_data["AnyHealthcare"] = 1 if opcion == "Sí" else 0  # Convertimos a 1 o 0

        opcion = st.selectbox(feature_labels["NoDocbcCost"], ["No", "Sí"])
        input_data["NoDocbcCost"] = 1 if opcion == "Sí" else 0  # Convertimos a 1 o 0

  
    # Sección 4: Salud General
   
    st.subheader("Salud General")
    col1, col2 = st.columns(2)

    with col1: # Para salud general, usamos un deslizador del 1 al 5
        input_data["GenHlth"] = st.slider(feature_labels["GenHlth"], 1, 5, 3)

        input_data["MentHlth"] = st.number_input(feature_labels["MentHlth"], 0, 30, 0, step=1)

    with col2:
        input_data["PhysHlth"] = st.number_input(feature_labels["PhysHlth"], 0, 30, 0, step=1)


    # Sección 5: Datos Personales
    
    st.subheader("Datos Personales")
    col1, col2, col3 = st.columns(3)

    with col1:
        sexo = st.selectbox(feature_labels["Sex"], ["Femenino", "Masculino"])
        input_data["Sex"] = 1 if sexo == "Masculino" else 0

    with col2:
        input_data["Age"] = st.slider(feature_labels["Age"], 18, 100, 50)
        opciones_educacion = [
            "1. Sin nivel educativo", "2. Pre-básica", "3. Básica", "4. Media",
            "5. Universitario incompleto", "6. Universitario completo"
        ]
        seleccion = st.selectbox(feature_labels["Education"], opciones_educacion)
        input_data["Education"] = int(seleccion.split(".")[0]) # Solo guardamos el número

    with col3:
        opciones_ingreso = [
            "1. 0 – 10,000", "2. 10,001 – 25,000", "3. 25,001 – 35,000",
            "4. 35,001 – 50,000", "5. 50,001 – 75,000", "6. 75,001 – 100,000",
            "7. 100,001 – 150,000", "8. Más de 150,000"
        ]
        seleccion = st.selectbox(feature_labels["Income"], opciones_ingreso)
        input_data["Income"] = int(seleccion.split(".")[0]) # Solo guardamos el número

    # Botón de envío del formulario
    submitted = st.form_submit_button("Evaluar Riesgo de Diabetes")


# RESULTADO DE LA PREDICCIÓN


if submitted:
    # Creamos un DataFrame con los datos en el orden exacto
    X_user = pd.DataFrame([input_data])[features]

    # Escalamos los datos para que el modelo los entienda
    X_user_scaled = scaler.transform(X_user)

    # Calculamos probabilidad de riesgo
    proba = model.predict_proba(X_user_scaled)[0][1]

    # Decidimos si hay riesgo según el umbral
    pred = 1 if proba >= umbral_optimo else 0

    st.divider()
    st.subheader("Resultado de la Evaluación")

    if pred == 1:
        st.error("⚠️ **Riesgo Elevado de Diabetes**")
        st.progress(proba, text=f"Probabilidad estimada: {proba:.1%}")
        st.warning("""
        **Recomendaciones:**
        - Consulte con su médico para una evaluación completa
        - Implemente cambios en dieta y ejercicio
        - Monitoree sus niveles de glucosa regularmente
        - Realice chequeos médicos periódicos
        """)
    else:
        st.success("✅ **Riesgo Bajo de Diabetes**")
        st.progress(proba, text=f"Probabilidad estimada: {proba:.1%}")
        st.info("""
        **Recomendaciones:**
        - Mantenga hábitos saludables de alimentación
        - Realice actividad física regularmente
        - Controle sus chequeos médicos anuales
        - Continúe con sus buenos hábitos de salud
        """)





