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

# ─────────────────────────────
# ░▒▓█ EN-TÊTE VISUEL █▓▒░
# ─────────────────────────────
col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])
with col_logo1:
    st.image("Shell.png", width=90)
with col_title:
    st.markdown("<h1 style='text-align: center; color: #3bce6c; font-size: 42px;'>Application de Prédiction du Churn</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #0f451f;'>Clients du Programme Fidélité Vivo Energy Maroc – Shell Licensee</h4>", unsafe_allow_html=True)
with col_logo2:
    st.image("Vivo.png", width=90)

# ─────────────────────────────
# ░▒▓█ INTRODUCTION ET GUIDE █▓▒░
# ─────────────────────────────
st.markdown("""
<div style='text-align: justify; font-size: 16px; line-height: 1.7;'>
Bienvenue sur notre outil d'aide à la décision basé sur l’intelligence artificielle. Cette application vous permet de <b>prédire le risque de départ d’un client</b> à partir de son historique de fidélité chez Vivo Energy Shell Maroc.
<br><br>
📦 <b>Données utilisées :</b><br>
<i>Les données synthétiques peuvent être définies comme des informations annotées artificiellement. Elles sont générées par des algorithmes ou des simulations informatiques, et nous les utilisons ici pour éviter toute diffusion de données confidentielles, tout en conservant les mêmes variables, structures et échelles que les données réelles de l’entreprise.</i>
<br><br>
🔍 <b>Fonctionnalités proposées :</b>
<ul>
  <li>Visualisation de KPI's stratégiques</li>
  <li>Choix et entraînement de modèles prédictifs</li>
  <li>Évaluation automatique des performances</li>
  <li>Simulation d’une prédiction client personnalisée</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# ░▒▓█ KPIs █▓▒░
# ─────────────────────────────
df = load_and_prepare_data()
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div style='background-color:#3bce6c;padding:20px;border-radius:10px;text-align:center'><h4 style='color:white;'>📊 Clients analysés</h4><h2 style='color:white;'>{len(df)}</h2></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div style='background-color:#3bce6c;padding:20px;border-radius:10px;text-align:center'><h4 style='color:white;'>💔 Taux de Churn</h4><h2 style='color:white;'>{round(df['Churned'].mean()*100, 2)}%</h2></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div style='background-color:#3bce6c;padding:20px;border-radius:10px;text-align:center'><h4 style='color:white;'>⭐ Points moyens</h4><h2 style='color:white;'>{round(df['Total_Points'].mean(), 0)}</h2></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div style='background-color:#3bce6c;padding:20px;border-radius:10px;text-align:center'><h4 style='color:white;'>🔁 Ratio de fidélité</h4><h2 style='color:white;'>{round(df['Loyalty_Ratio'].mean(), 2)}</h2></div>", unsafe_allow_html=True)

# ─────────────────────────────
# ░▒▓█ CHOIX DU MODÈLE █▓▒░
# ─────────────────────────────
st.markdown("---")
st.subheader("🔧 Choix du modèle d’apprentissage automatique")
model_choice = st.selectbox("Sélectionnez un modèle :", ["Random Forest", "Logistic Regression", "KNN"])

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


