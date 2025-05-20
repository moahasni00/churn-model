import streamlit as st
import pandas as pd
from model_utils import load_and_prepare_data, train_model, plot_confusion

st.set_page_config(page_title='Prédiction de Churn - Shell Loyalty', layout='wide')
st.title("📊 Application de Prédiction de Churn - Clients Fidélité Shell")

# Chargement des données
df = load_and_prepare_data()

# Statistiques descriptives
if st.checkbox("📌 Afficher les statistiques descriptives"):
    st.subheader("🔍 Vue d’ensemble des données")
    st.dataframe(df.describe())

# Entraînement du modèle
if st.button("🔁 Entraîner le modèle de Machine Learning"):
    model, scaler, metrics = train_model(df)
    st.success("✅ Le modèle a été entraîné avec succès.")

    # Résultats d’évaluation
    st.subheader("📈 Évaluation du Modèle")
    st.write(f"**Exactitude (Accuracy) :** {metrics['accuracy']:.2f}")
    st.write(f"**Score ROC AUC :** {metrics['roc_auc']:.2f}")

    st.subheader("📊 Rapport de Classification")
    st.dataframe(pd.DataFrame(metrics['classification']).transpose())

    st.subheader("🧱 Matrice de Confusion")
    fig = plot_confusion(metrics['conf_matrix'])
    st.pyplot(fig)

# Prédiction manuelle
st.subheader("🔮 Faire une prédiction personnalisée")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    total_points = col1.number_input("Points totaux", min_value=0, max_value=50000, value=15000)
    points_redeemed = col2.number_input("Points utilisés", min_value=0, max_value=50000, value=8000)
    fuel_type = col3.selectbox("Type de carburant", ["V-Power Sans Plomb", "FuelSave", "V-Power Diesel", "V-Power"])

    lubricant = col1.slider("Montant lubrifiants (MAD)", 0, 500, 50)
    shop = col2.slider("Achats magasin (MAD)", 0, 1000, 100)
    resto = col3.slider("Restaurant La Pause (MAD)", 0, 500, 50)
    butagaz = col1.slider("Nombre de bouteilles Butagaz", 0, 5, 1)

    submit = st.form_submit_button("🎯 Prédire le churn")
    if submit:
        loyalty_ratio = points_redeemed / total_points if total_points else 0
        est_points = (total_points - points_redeemed) * (100 / 20 if fuel_type == "FuelSave" else 100 / 40)
        earned = est_points + lubricant + shop + resto + butagaz * 100
        fuel_encoded = {"V-Power Sans Plomb": 0, "FuelSave": 1, "V-Power Diesel": 2, "V-Power": 3}[fuel_type]

        input_data = pd.DataFrame([[total_points, fuel_encoded, lubricant, shop, resto,
                                    butagaz, points_redeemed, earned, loyalty_ratio]],
                                  columns=df.drop("Churned", axis=1).columns)

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.markdown(f"### ✅ Résultat : {'❌ Client à risque de churn' if prediction else '✔️ Client fidèle'}")
        st.markdown(f"### 🔢 Probabilité estimée de churn : **{prob*100:.2f}%**")

