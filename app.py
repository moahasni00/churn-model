import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_and_prepare_data, plot_confusion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

st.set_page_config(page_title="Churn - Fidélité Shell", layout="wide")

# ────────────────────────────────
# ░▒▓█ LOGOS DANS L’EN-TÊTE █▓▒░
# ────────────────────────────────
col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])
with col_logo1:
    st.image("Shell.png", width=90)
with col_title:
    st.title("📊 Application de Prédiction de Churn - Clients Fidélité Shell")
with col_logo2:
    st.image("Vivo.png", width=90)

# ────────────────────────────────
# ░▒▓█ CHARGEMENT DES DONNÉES █▓▒░
# ────────────────────────────────
df = load_and_prepare_data()

# ────────────────────────────────
# ░▒▓█ KPI RAPIDES █▓▒░
# ────────────────────────────────
if st.checkbox("📌 Aperçu des données & KPI"):
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total clients", f"{len(df)}")
    col2.metric("🔁 % Churn", f"{round(df['Churned'].mean()*100,2)}%")
    col3.metric("⭐ Points moyens", f"{round(df['Total_Points'].mean(),0)}")

    st.subheader("📍 Premier aperçu des données")
    st.dataframe(df.head())

if st.checkbox("📈 Statistiques globales"):
    st.subheader("📊 Statistiques")
    st.dataframe(df.describe())

# ────────────────────────────────
# ░▒▓█ CHOIX DU MODÈLE █▓▒░
# ────────────────────────────────
st.sidebar.header("🔧 Choix du Modèle ML")
model_choice = st.sidebar.selectbox("Modèle", ["Random Forest", "Logistic Regression", "KNN"])

model = None
scaler = None

# ────────────────────────────────
# ░▒▓█ ENTRAÎNEMENT █▓▒░
# ────────────────────────────────
if st.button("🎯 Entraîner le modèle ML"):
    X = df.drop("Churned", axis=1)
    y = df["Churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.success("✅ Modèle entraîné avec succès")
    st.write(f"**Exactitude (Accuracy)** : {acc:.2f}")
    st.write(f"**AUC Score** : {roc:.2f}")

    st.subheader("📘 Interprétation des résultats :")
    if acc > 0.85:
        st.markdown("✅ **Le modèle est fiable et peut être utilisé pour la prise de décision**, car il atteint une exactitude supérieure à 85%, ce qui signifie que plus de 8 prédictions sur 10 sont correctes.")
    elif acc > 0.7:
        st.markdown("🟡 **Modèle acceptable**, mais une amélioration est envisageable pour une meilleure décision.")
    else:
        st.markdown("❌ **Modèle non recommandé pour les décisions importantes.**")

    st.subheader("📄 Rapport de Classification")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📊 Matrice de confusion")
    fig = plot_confusion(cm)
    st.pyplot(fig)

    if model_choice == "Random Forest":
        st.subheader("📌 Importance des variables")
        importance = model.feature_importances_
        fig2, ax = plt.subplots(figsize=(6, 4))
        pd.Series(importance, index=X.columns).sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig2)

    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# ────────────────────────────────
# ░▒▓█ PRÉDICTION MANUELLE █▓▒░
# ────────────────────────────────
st.subheader("🔮 Prédiction personnalisée")
with st.form("form_predict"):
    col1, col2, col3 = st.columns(3)

    total_points = col1.number_input("Points totaux", 0, 50000, 15000)
    points_redeemed = col2.number_input("Points utilisés", 0, 50000, 7000)
    fuel_type = col3.selectbox("Type de carburant", ["V-Power Sans Plomb", "FuelSave", "V-Power Diesel", "V-Power"])

    lubricant = col1.slider("Montant lubrifiants (MAD)", 0, 500, 100, help="min=0 / max=500")
    shop = col2.slider("Achats magasin (MAD)", 0, 1000, 200, help="min=0 / max=1000")
    resto = col3.slider("Restaurant La Pause (MAD)", 0, 500, 50, help="min=0 / max=500")
    butagaz = col1.slider("Nombre de bouteilles Butagaz", 0, 5, 1)

    submitted = st.form_submit_button("🚀 Prédire le churn")
    if submitted:
        if "model" in st.session_state:
            loyalty_ratio = points_redeemed / total_points if total_points else 0
            est_points = (total_points - points_redeemed) * (100 / 20 if fuel_type == "FuelSave" else 100 / 40)
            earned = est_points + lubricant + shop + resto + butagaz * 100
            fuel_encoded = {"V-Power Sans Plomb": 0, "FuelSave": 1, "V-Power Diesel": 2, "V-Power": 3}[fuel_type]

            input_data = pd.DataFrame([[total_points, fuel_encoded, lubricant, shop, resto,
                                        butagaz, points_redeemed, earned, loyalty_ratio]],
                                      columns=df.drop("Churned", axis=1).columns)

            scaled = st.session_state["scaler"].transform(input_data)
            result = st.session_state["model"].predict(scaled)[0]
            prob = st.session_state["model"].predict_proba(scaled)[0][1]

            st.markdown(f"### Résultat : {'❌ Client à risque' if result else '✔️ Client fidèle'}")
            st.markdown(f"### Probabilité : **{prob*100:.2f}%**")
        else:
            st.warning("Veuillez d’abord entraîner le modèle.")

# ────────────────────────────────
# ░▒▓█ WATERMARK █▓▒░
# ────────────────────────────────
st.markdown("""<hr style="border: 1px solid #ddd;">
<div style='text-align: center; font-size: 14px;'>
    <i>Réalisée par : <b>Oumaima Zaz</b> - Université Hassan I – Master en Marketing et action commerciale</i>
</div>""", unsafe_allow_html=True)

