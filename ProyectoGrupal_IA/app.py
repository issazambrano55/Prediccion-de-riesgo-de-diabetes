import streamlit as st  # Esto es para hacer la página web interactiva
import pandas as pd      # Para trabajar con tablas de datos (dataframes)
import joblib            # Para cargar modelos de IA que ya entrenamos

# Cargar el modelo de predicción de diabetes, el escalador de datos y el umbral óptimo
model = joblib.load("modelo_diabetes.pkl")  # Nuestro modelo entrenado
scaler = joblib.load("scaler.pkl")         # Para normalizar los datos (hacerlos comparables)
with open("umbral_optimo.txt", "r") as f:  
    umbral_optimo = float(f.read())        # El valor que usamos para decidir si hay riesgo

# Estas son las columnas que nuestro modelo necesita
features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Aquí damos nombres más amigables a cada columna para mostrar al usuario
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
    "AnyHealthcare": "Acceso a atención médica",
    "NoDocbcCost": "No fue al médico por costo",
    "GenHlth": "Salud general (1=Excelente, 5=Mala)",
    "MentHlth": "Días de mala salud mental (últimos 30 días)",
    "PhysHlth": "Días de mala salud física (últimos 30 días)",
    "DiffWalk": "Dificultad para caminar",
    "Sex": "Sexo",
    "Age": "Edad",
    "Education": "Nivel educativo",
    "Income": "Ingreso anual"
}

# Aquí empezamos el formulario donde el usuario va a ingresar sus datos
with st.form("formulario_entrada"):
    st.title("Predicción de Riesgo de Diabetes")  # Título de la app
    st.write("Ingrese los datos del paciente:")   # Subtítulo

    input_data = {}  # Creamos un diccionario vacío donde vamos a guardar los datos

    # Pedimos peso y altura para calcular el BMI
    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0)
    altura = st.number_input("Altura (cm)", min_value=100.0, max_value=250.0, value=170.0)
    altura_m = altura / 100
    bmi_calc = round(peso / (altura_m ** 2), 2)  # Fórmula del BMI

    # Ahora vamos a recorrer todas las características que necesita el modelo
    for feature in features:
        label = feature_labels[feature]  # Tomamos el nombre amigable para mostrar

        if feature == "BMI":
            # Si es BMI, usamos el valor que calculamos
            input_data[feature] = st.number_input(label, value=bmi_calc, format="%.2f")

        elif feature in [
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "DiffWalk"
        ]:
            # Para las preguntas de "Sí/No"
            opcion = st.selectbox(label, options=["No", "Sí"])
            input_data[feature] = 1 if opcion == "Sí" else 0  # Convertimos a 1 o 0

        elif feature == "Sex":
            sexo = st.selectbox(label, options=["F", "M"])
            input_data[feature] = 1 if sexo == "M" else 0

        elif feature == "GenHlth":
            # Para salud general, usamos un deslizador del 1 al 5
            input_data[feature] = st.slider(label, min_value=1, max_value=5, value=3)

        elif feature == "Education":
            # Para nivel educativo
            opciones_educacion = [
                "1. Sin nivel educativo",
                "2. Pre-básica",
                "3. Básica",
                "4. Media",
                "5. Universitario incompleto",
                "6. Universitario completo"
            ]
            seleccion = st.selectbox(label, opciones_educacion)
            input_data[feature] = int(seleccion.split(".")[0])  # Solo guardamos el número

        elif feature == "Income":
            # Para ingreso anual
            opciones_ingreso = [
                "1. 0 – 10,000",
                "2. 10,001 – 25,000",
                "3. 25,001 – 35,000",
                "4. 35,001 – 50,000",
                "5. 50,001 – 75,000",
                "6. 75,001 – 100,000",
                "7. 100,001 – 150,000",
                "8. Más de 150,000"
            ]
            seleccion = st.selectbox(label, opciones_ingreso)
            input_data[feature] = int(seleccion.split(".")[0])

        elif feature == "Age":
            input_data[feature] = st.slider(label, min_value=18, max_value=100, value=50)

        elif feature in ["MentHlth", "PhysHlth"]:
            # Para los días de mala salud
            input_data[feature] = st.number_input(label, min_value=0.0, max_value=30.0, value=0.0)

    # Botón para enviar los datos
    submitted = st.form_submit_button("Predecir")

# Si el usuario hace clic en "Predecir"
if submitted:
    # Creamos un dataframe con los datos del usuario
    X_user = pd.DataFrame([input_data])[features]
    # Normalizamos los datos
    X_user_scaled = scaler.transform(X_user)
    # Probabilidad de que tenga prediabetes/diabetes
    proba = model.predict_proba(X_user_scaled)[0][1]
    # Predicción final según el umbral
    pred = 1 if proba >= umbral_optimo else 0

    # Mostramos el resultado
    st.subheader("Resultado de la predicción")
    if pred == 1:
        st.error(f"⚠ Riesgo detectado de prediabetes o diabetes (probabilidad: {proba:.2%})")
    else:
        st.success(f"✅ No hay riesgo detectado (probabilidad: {proba:.2%})")


