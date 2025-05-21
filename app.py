import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import load_and_prepare_data, plot_confusion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

st.set_page_config(page_title="Churn - Fidélité Shell", layout="wide")

# ────── EN-TÊTE ──────
col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])
with col_logo1:
    st.image("Shell.png", width=85)
with col_title:
    st.markdown("<h1 style='text-align: center; color: #3bce6c; font-size: 42px;'>Application de Prédiction du Churn</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #0f451f;'>Clients du Programme Fidélité Vivo Energy Maroc – Shell Licensee</h4>", unsafe_allow_html=True)
with col_logo2:
    st.image("Vivo.png", width=110)

# ────── INTRO ──────
st.markdown("""
<div style='text-align: justify; font-size: 16px; line-height: 1.7;'>
Bienvenue sur notre outil d'analyse prédictive du comportement client. Cette application vous permet d’<b>estimer le risque de départ d’un client</b> à partir de son historique de fidélité dans le cadre du programme Vivo Energy Maroc - Shell.
<br><br>
📦 <b>Données utilisées :</b><br>
<i>Les données synthétiques peuvent être définies comme des informations annotées artificiellement. Elles sont générées par des algorithmes ou des simulations informatiques, et nous les utilisons ici pour éviter toute diffusion de données confidentielles, tout en conservant les mêmes variables, structures et échelles que les données réelles de l’entreprise.</i>
</div>
""", unsafe_allow_html=True)

# ────── DONNÉES ──────
df = load_and_prepare_data()

# ────── CHOIX DU MODÈLE ──────
st.markdown("---")
st.subheader("🔧 Choix du modèle d’apprentissage automatique")
model_choice = st.selectbox("Sélectionnez un modèle :", ["Random Forest", "Logistic Regression", "KNN"])

# ────── KPI & APERÇU ──────
if st.checkbox("📌 Aperçu des données & KPI"):
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div style='background-color:#3bce6c;padding:6px 8px;border-radius:3px; border:2px solid black;text-align:center'><h6 style='color:white;'>Clients analysés</h6><h5 style='color:white;margin:0'>{len(df)}</h5></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div style='background-color:#3bce6c;padding:6px 8px;border-radius:3px; border:2px solid black;text-align:center'><h6 style='color:white;'>Taux de churn</h6><h5 style='color:white;margin:0'>{round(df['Churned'].mean()*100, 2)}%</h5></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div style='background-color:#3bce6c;padding:6px 8px;border-radius:3px; border:2px solid black;text-align:center'><h6 style='color:white;'>Points moyens</h6><h5 style='color:white;margin:0'>{round(df['Total_Points'].mean(), 0)}</h5></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div style='background-color:#3bce6c;padding:6px 8px;border-radius:3px; border:2px solid black;text-align:center'><h6 style='color:white;'>Ratio fidélité</h6><h5 style='color:white;margin:0'>{round(df['Loyalty_Ratio'].mean(), 2)}</h5></div>", unsafe_allow_html=True)
    st.subheader("📍 Premier aperçu des données")
    st.dataframe(df.head())

# ────── ENTRAÎNEMENT ──────
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

    st.success("✅ Modèle entraîné avec succès")
    st.write(f"**Exactitude (Accuracy)** : {acc:.2f}")
    st.write(f"**AUC Score** : {roc:.2f}")

    st.subheader("📘 Interprétation :")
    if acc > 0.85:
        st.markdown("✅ **Modèle fiable pour la prise de décision.**")
    elif acc > 0.7:
        st.markdown("🟡 **Modèle acceptable.**")
    else:
        st.markdown("❌ **Modèle peu performant.**")

    st.subheader("📄 Rapport de classification")
    st.dataframe(pd.DataFrame(report).transpose())

    if model_choice == "Random Forest":
        st.subheader("📌 Importance des variables")
        fig2, ax = plt.subplots(figsize=(6, 4))
        pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig2)

    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# ────── ANALYSE EXPLORATOIRE ──────
st.markdown("---")
st.subheader("📊 Analyse exploratoire des variables")

colv1, colv2 = st.columns(2)

with colv1:
    st.markdown("**Répartition du churn**")
    fig, ax = plt.subplots()
    df['Churned'].value_counts().plot(kind='bar', color=['#0f451f', '#3bce6c'], ax=ax)
    ax.set_xticklabels(['Fidèle', 'Churné'], rotation=0)
    st.pyplot(fig)

with colv2:
    st.markdown("**Ratio de fidélité selon le statut**")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x="Churned", y="Loyalty_Ratio", palette=["#0f451f", "#3bce6c"], ax=ax2)
    ax2.set_xticklabels(['Fidèle', 'Churné'])
    st.pyplot(fig2)

st.markdown("**Distribution des points fidélité**")
fig3, ax3 = plt.subplots()
sns.histplot(df['Total_Points'], bins=30, color='#0f451f', ax=ax3)
ax3.set_title("Histogramme des Points Totaux")
ax3.set_xlabel("Points")
st.pyplot(fig3)

st.markdown("**Répartition du churn selon le type de carburant**")
fig4, ax4 = plt.subplots()
df.groupby("Fuel_Type")["Churned"].mean().plot(kind="bar", color="#3bce6c", ax=ax4)
ax4.set_ylabel("Taux de churn")
ax4.set_title("Type de carburant vs Taux de churn")
st.pyplot(fig4)

st.markdown("**Matrice de corrélation**")
fig5, ax5 = plt.subplots(figsize=(8, 6))
corr = df.drop(columns="Fuel_Type").corr()
sns.heatmap(corr, annot=True, cmap="Greens", ax=ax5)
st.pyplot(fig5)

# ────── PRÉDICTION ──────
st.subheader("🔮 Prédiction personnalisée")
with st.form("form_predict"):
    col1, col2, col3 = st.columns(3)

    total_points = col1.number_input("Points totaux", 
        min_value=int(df['Total_Points'].min()), 
        max_value=int(df['Total_Points'].max()), 
        value=15000)
    col1.caption(f"Min: {df['Total_Points'].min()} - Max: {df['Total_Points'].max()}")

    points_redeemed = col2.number_input("Points utilisés", 
        min_value=0, 
        max_value=int(df['Points_Redeemed'].max()), 
        value=7000)
    col2.caption(f"Min: 0 - Max: {df['Points_Redeemed'].max()}")

    fuel_type = col3.selectbox("Type de carburant", ["V-Power Sans Plomb", "FuelSave", "V-Power Diesel", "V-Power"])
    col3.caption("Types disponibles : 4")

    lubricant = col1.slider("Montant lubrifiants (MAD)", 0, 500, 100)
    shop = col2.slider("Achats magasin (MAD)", 0, 1000, 200)
    resto = col3.slider("Restaurant La Pause (MAD)", 0, 500, 50)
    butagaz = col1.slider("Nombre de bouteilles Butagaz", 0, 5, 1)

    submitted = st.form_submit_button("🚀 Prédire le churn")
    if submitted and "model" in st.session_state:
        loyalty_ratio = points_redeemed / total_points if total_points else 0
        est_points = (total_points - points_redeemed) * (100 / 20 if fuel_type == "FuelSave" else 100 / 40)
        earned = est_points + lubricant + shop + resto + butagaz * 100
        fuel_encoded = {"V-Power Sans Plomb": 0, "FuelSave": 1, "V-Power Diesel": 2, "V-Power": 3}[fuel_type]

        input_data = pd.DataFrame([[total_points, fuel_encoded, lubricant, shop, resto, butagaz, points_redeemed, earned, loyalty_ratio]], columns=df.drop("Churned", axis=1).columns)

        scaled = st.session_state["scaler"].transform(input_data)
        result = st.session_state["model"].predict(scaled)[0]
        prob = st.session_state["model"].predict_proba(scaled)[0][1]

        st.markdown(f"### Résultat : {'❌ Client à risque' if result else '✔️ Client fidèle'}")
        st.markdown(f"### Probabilité : **{prob*100:.2f}%**")
    elif submitted:
        st.warning("Veuillez d’abord entraîner le modèle.")

# 🎨 STYLE SLIDER EN VERT (#0f451f) UNIQUEMENT BARRE & CURSEUR
st.markdown("""
<style>
/* Barre de fond du slider */
div[data-baseweb="slider"] > div > div:nth-child(2) > div {
    background: #0f451f !important;
}

/* Curseur rond du slider */
div[data-baseweb="slider"] span[role="slider"] {
    background-color: #0f451f !important;
    border-color: #0f451f !important;
}
</style>
""", unsafe_allow_html=True)

# ────── WATERMARK ──────
st.markdown("""<hr style="border: 1px solid #ddd;">
<div style='text-align: center; font-size: 14px;'>
    <i>Réalisée par : <b>Oumaima Zaz</b> - Université Hassan I – Master en Marketing et action commerciale</i>
</div>""", unsafe_allow_html=True)
