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

# Page configuration
st.set_page_config(page_title="Churn - Fidélité Shell", layout="wide")

# ────── EN-TÊTE ──────
col_logo1, col_title, col_logo2 = st.columns([1, 6, 1])
with col_logo1:
    st.image("Shell.png", width=85)
with col_title:
    st.markdown("<h1 style='text-align: center; color: #3bce6c; font-size: 42px;'>Application de Prédiction du Churn</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0f451f; font-size: 24px;'>Clients du Programme Fidélité Vivo Energy Maroc – Shell Licensee</h3>", unsafe_allow_html=True)
with col_logo2:
    st.image("Vivo.png", width=110)

# ────── INTRO ──────
st.markdown("""
<div style='text-align: justify; font-size: 16px; line-height: 1.7; margin-bottom: 20px;'>
Bienvenue sur notre outil d'analyse prédictive du comportement client. Cette application vous permet d'<b>estimer le risque de départ d'un client</b> à partir de son historique de fidélité dans le cadre du programme Vivo Energy Maroc - Shell.
</div>
""", unsafe_allow_html=True)

# ────── À PROPOS DES DONNÉES ──────
with st.expander("ℹ️ À propos des données"):
    st.markdown("""
    <div style='text-align: justify; font-size: 16px; line-height: 1.7;'>
    📦 <b>Données utilisées :</b><br>
    <i>Les données synthétiques peuvent être définies comme des informations annotées artificiellement. Elles sont générées par des algorithmes ou des simulations informatiques, et nous les utilisons ici pour éviter toute diffusion de données confidentielles, tout en conservant les mêmes variables, structures et échelles que les données réelles de l'entreprise.</i>
    </div>
    """, unsafe_allow_html=True)

# ────── SÉPARATEUR VISUEL ──────
st.markdown("""
<div style='height: 3px; background: linear-gradient(90deg, #3bce6c, #0f451f); margin: 25px 0 15px 0;'></div>
""", unsafe_allow_html=True)

# ────── DONNÉES ──────
df = load_and_prepare_data()

# ────── KPI & APERÇU ──────
st.markdown("""
<div style="margin-top: 30px;">
    <h3 style="color: #0f451f; font-size: 22px;">📊 Tableau de bord</h3>
</div>
""", unsafe_allow_html=True)

# CSS pour les cartes KPI
st.markdown("""
<style>
.kpi-card {
    background-color: #3bce6c;
    border-radius: 12px;
    border: 3px solid #0f451f;
    padding: 12px 15px;
    text-align: center;
    height: 100%;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}
.kpi-title {
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 8px;
}
.kpi-value {
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin: 0;
}
.section-container {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# ────── KPI & APERÇU SECTION ──────
# Utilisation d'un expander ou checkbox personnalisé pour l'aperçu des données et KPI
show_kpi = st.checkbox("📌 Aperçu des données & KPI", value=True)

if show_kpi:
    # Affichage des KPIs
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Clients analysés</div>
            <div class="kpi-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Taux de churn</div>
            <div class="kpi-value">{round(df['Churned'].mean()*100, 2)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Points moyens</div>
            <div class="kpi-value">{round(df['Total_Points'].mean(), 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Ratio fidélité</div>
            <div class="kpi-value">{round(df['Loyalty_Ratio'].mean(), 2)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Premier aperçu des données
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<span class='apercu-label'>📍 Premier aperçu des données</span>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# ────── CHOIX DU MODÈLE ──────
st.markdown("""
<div style='height: 3px; background: linear-gradient(90deg, #3bce6c, #0f451f); margin: 30px 0 15px 0;'></div>
<h3 style="color: #0f451f; font-size: 22px; margin-bottom: 15px;">🔧 Choix du modèle d'apprentissage automatique</h3>
""", unsafe_allow_html=True)
model_choice = st.selectbox("Sélectionnez un modèle :", ["Random Forest", "Logistic Regression", "KNN"])

# ────── ENTRAÎNEMENT ──────
train_btn = st.button("🎯 Entraîner le modèle ML", type="primary")
if train_btn:
    with st.spinner("Entraînement en cours..."):
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
    
    # Affichage des résultats dans un conteneur avec style
    st.markdown("""
    <div class="section-container">
        <h3 style="color: #0f451f; font-size: 22px; margin-bottom: 15px;">📊 Résultats du modèle</h3>
    """, unsafe_allow_html=True)
    
    st.success("✅ Modèle entraîné avec succès")
    
    # Métriques de performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Exactitude (Accuracy)", value=f"{acc:.2f}")
    with col2:
        st.metric(label="AUC Score", value=f"{roc:.2f}")

    # Interprétation
    st.subheader("📘 Interprétation :")
    if acc > 0.85:
        st.markdown("✅ **Modèle fiable pour la prise de décision.**")
    elif acc > 0.7:
        st.markdown("🟡 **Modèle acceptable.**")
    else:
        st.markdown("❌ **Modèle peu performant.**")

    # Rapport de classification
    st.subheader("📄 Rapport de classification")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # Importance des variables pour Random Forest
    if model_choice == "Random Forest":
        st.subheader("📌 Importance des variables")
        fig2, ax = plt.subplots(figsize=(10, 6))
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        feature_importance.plot(kind='barh', ax=ax, color='#3bce6c')
        ax.set_title('Importance des variables', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# ────── PRÉDICTION ──────
st.markdown("""
<div style='height: 3px; background: linear-gradient(90deg, #3bce6c, #0f451f); margin: 30px 0 15px 0;'></div>
<h3 style='color: #0f451f; font-size: 22px; margin-bottom: 15px;'>🔮 Prédiction personnalisée</h3>
""", unsafe_allow_html=True)

with st.form("form_predict"):
    # Disposition améliorée des formulaires
    st.markdown("""
    <style>
    .form-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #dee2e6;
    }
    </style>
    <div class="form-container">
    """, unsafe_allow_html=True)
    
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

    submitted = st.form_submit_button("🚀 Prédire le churn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submitted and "model" in st.session_state:
        loyalty_ratio = points_redeemed / total_points if total_points else 0
        est_points = (total_points - points_redeemed) * (100 / 20 if fuel_type == "FuelSave" else 100 / 40)
        earned = est_points + lubricant + shop + resto + butagaz * 100
        fuel_encoded = {"V-Power Sans Plomb": 0, "FuelSave": 1, "V-Power Diesel": 2, "V-Power": 3}[fuel_type]

        input_data = pd.DataFrame([[total_points, fuel_encoded, lubricant, shop, resto, butagaz, points_redeemed, earned, loyalty_ratio]], 
                                 columns=df.drop("Churned", axis=1).columns)

        scaled = st.session_state["scaler"].transform(input_data)
        result = st.session_state["model"].predict(scaled)[0]
        prob = st.session_state["model"].predict_proba(scaled)[0][1]

        # Affichage amélioré des résultats de prédiction
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 12px; padding: 20px; margin-top: 20px; border: 1px solid #dee2e6;">
            <h3 style="text-align: center; margin-bottom: 15px;">Résultat de la prédiction</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            status = "❌ Client à risque" if result else "✅ Client fidèle"
            color = "#dc3545" if result else "#28a745"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background-color: {color}; border-radius: 12px; color: white;">
                <h2 style="margin: 0;">{status}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Création d'une jauge pour la probabilité
            risk_level = "Élevé" if prob > 0.7 else "Moyen" if prob > 0.3 else "Faible"
            risk_color = "#dc3545" if prob > 0.7 else "#ffc107" if prob > 0.3 else "#28a745"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <h4>Probabilité de départ</h4>
                <h2 style="color: {risk_color};">{prob*100:.2f}%</h2>
                <p>Niveau de risque: <strong style="color: {risk_color};">{risk_level}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif submitted:
        st.warning("⚠️ Veuillez d'abord entraîner le modèle.")

# ────── STYLE AMÉLIORÉ ──────
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

/* Style pour les boutons principaux */
button[kind="primary"] {
    background-color: #3bce6c !important;
    border-color: #0f451f !important;
    color: white !important;
}

button[kind="primary"]:hover {
    background-color: #2da559 !important;
}

/* Sélection */
.stSelectbox div[data-baseweb="select"] > div {
    border-color: #0f451f !important;
}

/* Entête des sections */
h3 {
    color: #0f451f !important;
    border-bottom: 2px solid #3bce6c;
    padding-bottom: 8px;
    margin-bottom: 20px !important;
    font-size: 22px !important;
}

/* Premier aperçu étiquette */
.apercu-label {
    font-size: 22px !important;
    color: #0f451f !important;
    margin-top: 20px !important;
    margin-bottom: 15px !important;
    display: inline-block !important;
    border-bottom: 2px solid #3bce6c;
    padding-bottom: 8px;
}

/* Style pour le checkbox d'aperçu */
.stCheckbox label {
    font-size: 18px !important;
    font-weight: bold !important;
    color: #0f451f !important;
}
</style>
""", unsafe_allow_html=True)

# ────── WATERMARK ──────
st.markdown("""
<div style='height: 3px; background: linear-gradient(90deg, #3bce6c, #0f451f); margin: 30px 0 15px 0;'></div>
<div style='text-align: center; font-size: 16px; padding: 15px; background-color: #f8f9fa; border-radius: 12px;'>
    <i>Réalisée par : <b>Oumaima Zaz</b> - Université Hassan I – Master en Marketing et action commerciale</i>
</div>""", unsafe_allow_html=True)
