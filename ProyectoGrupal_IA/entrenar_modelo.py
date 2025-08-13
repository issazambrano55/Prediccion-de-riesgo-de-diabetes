# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sb

# Librerías para dividir datos, escalar y entrenar el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

# Leer el archivo csv que tiene los datos de salud
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Ver si hay valores vacíos en las columnas
print("Valores nulos por columna:\n", df.isna().sum())

# Crear una columna nueva que diga si tiene riesgo de diabetes (1) o no (0)
# Si Diabetes_012 es mayor que 0 significa que tiene prediabetes o diabetes
df['DiabetesBinary'] = np.where(df['Diabetes_012'] > 0, 1, 0)

# Hacer gráficas para ver cómo se distribuyen las clases
sb.countplot(x="Diabetes_012", data=df); sb.despine()
sb.countplot(x="DiabetesBinary", data=df)

# Ver la correlación entre las columnas (solo las numéricas)
corr = df.corr(numeric_only=True)
sb.heatmap(corr, cmap="coolwarm", annot=False)

# X son las columnas con las variables que vamos a usar para predecir
# y es la columna con la respuesta (si tiene riesgo o no)
X = df.drop(columns=["Diabetes_012", "DiabetesBinary"])
y = df["DiabetesBinary"]

# Dividir los datos en entrenamiento (80%) y prueba (20%)
# stratify=y sirve para que las clases queden balanceadas en los conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Dividir el conjunto de entrenamiento en entrenamiento (80%) y validación (20%)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Escalar los datos (poner todas las columnas en la misma escala)
scaler = StandardScaler()
X_tr_scaled  = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión logística
# max_iter es cuántas veces intenta aprender
# class_weight='balanced' ayuda cuando hay clases desbalanceadas
# C=3.0 es un parámetro de regularización
model = LogisticRegression(max_iter=2000, class_weight='balanced', C=3.0)
model.fit(X_tr_scaled, y_tr)  # entrenar el modelo

# Calcular las probabilidades en el conjunto de validación
proba_val = model.predict_proba(X_val_scaled)[:, 1]

# Probar diferentes umbrales de decisión entre 0.2 y 0.8
thresholds = np.linspace(0.2, 0.8, 61)
accs = [accuracy_score(y_val, (proba_val >= t).astype(int)) for t in thresholds]

# Escoger el umbral que da la mejor precisión
best_t_idx = int(np.argmax(accs))
best_t = thresholds[best_t_idx]
best_val_acc = accs[best_t_idx]
print(f"Umbral óptimo en validación: {best_t:.3f} | Accuracy val: {best_val_acc:.4f}")

# Usar el umbral óptimo en el conjunto de prueba
proba_test = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (proba_test >= best_t).astype(int)

# Mostrar métricas en el conjunto de prueba
print("Accuracy TEST (con umbral óptimo):", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación (TEST):\n", classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión (TEST):\n", cm)

# Guardar el modelo entrenado y el escalador
import joblib
joblib.dump(model, "modelo_diabetes.pkl")
joblib.dump(scaler, "scaler.pkl")

# Guardar el umbral óptimo en un archivo de texto
with open("umbral_optimo.txt", "w") as f:
    f.write(str(best_t))
