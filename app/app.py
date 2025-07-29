import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Carregamento dos dados
@st.cache_data
def load_data():
    return pd.read_csv("data/datatran2025.csv", encoding="ISO-8859-1", delimiter=';')

df = load_data()
st.title("🔍 Classificação - Acidentes de Trânsito")

st.subheader("Prévia do Dataset")
st.dataframe(df.head())

# Seleção de variáveis categóricas e numéricas
st.sidebar.header("Configurações do Modelo")
target = st.sidebar.selectbox("Variável Alvo (classe Y)", df.select_dtypes(include="object").columns)

if target:
    feature_candidates = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns
else:
    feature_candidates = df.select_dtypes(include=[np.number]).columns

features = st.sidebar.multiselect("Variáveis preditoras (X)", feature_candidates)

if len(features) == 0:
    st.warning("Selecione pelo menos uma variável preditora.")
    st.stop()

# Preparação dos dados
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
acc = accuracy_score(y_test, y_pred)
st.subheader("🎯 Desempenho do Modelo")
st.write(f"**Acurácia:** {acc:.2%}")

# Relatório de Classificação
st.subheader("📋 Relatório de Classificação")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

# Matriz de Confusão
st.subheader("🔻 Matriz de Confusão")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)
