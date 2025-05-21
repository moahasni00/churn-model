import streamlit as st
import pandas as pd
from model_utils import load_and_prepare_data, plot_confusion
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title='Prédiction de Churn', layout='wide')
st.title("📊 Application de Prédiction de Churn - Clients Fidélité Shell")

# Charger les données
df = load_and_prepare_data()

# Choix du modèle
st.sidebar.header("🔧 Paramètres du Modèle")
model_choice = st.sidebar.selectbox("Choisir un modèle ML", ("Random Forest", "Logistic Regression", "KNN"))

# Affichage des données
if st.checkbox("📌 Aperçu des données"):
    st.subheader("🔍 Premier aperçu des données")
    st.dataframe(df.head())

if st.checkbox("📊 Statistiques descriptives"):
    st.subheader("📈 Statistiques globales")
    st.dataframe(df.describe())

# Initialisation
model = None
scaler = None

# Entraînement
if st.button("🔁 Entraîner le modèle de Machine Learning"):
    st.subheader(f"⚙️ Entraînement du modèle : {model_choice}")

    X = df.drop("Churned", axis=1)
    y = df["Churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choix du modèle
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Évaluation
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.success("✅ Modèle entraîné avec succès.")
    st.write(f"🎯 **Accuracy : {acc:.2f}**")
    st.write(f"📊 **ROC AUC : {roc:.2f}**")

    st.subheader("📘 Interprétation :")
    if acc > 0.85:
        st.markdown("✅ Le modèle est **fiable** et peut être utilisé pour la prise de décision.")
    elif acc > 0.7:
        st.markdown("🟡 Le modèle est **moyennement fiable**, à utiliser avec prudence.")
    else:
        st.markdown("❌ Le modèle est **peu fiable**, il est conseillé d’en tester un autre.")

    st.subheader("📄 Rapport de classification")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📊 Matrice de confusion")
    st.pyplot(plot_confusion(cm))

    if model_choice == "Random Forest":
        st.subheader("📌 Importance des variables")
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        pd.Series(importance, index=X.columns).sort_values().plot(kind='barh', ax=ax)
        st.pyplot(fig)

    # Stockage dans session_state pour la prédiction
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# Section de prédiction
st.subheader("🔮 Prédiction personnalisée")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    total_points = col1.number_input("Points totaux", min_value=0, max_value=50000, value=15000)
    points_redeemed = col2.number_input("Points utilisés", min_value=0, max_value=50000, value=8000)
    fuel_type = col3.selectbox("Type de carburant", ["V-Power Sans Plomb", "FuelSave", "V-Power Diesel", "V-Power"])

    lubricant = col1.slider("Montant lubrifiants (MAD)", 0, 500, 50)
    shop = col2.slider("Achats magasin (MAD)", 0, 1000, 100)
    resto = col3.slider("Restaurant La Pause (MAD)", 0, 500, 50)
    butagaz = col1.slider("Nombre de bouteilles Butagaz", 0, 5, 1)

    submitted = st.form_submit_button("🎯 Prédire le churn")
    if submitted:
        if "model" in st.session_state and "scaler" in st.session_state:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]

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
            st.markdown(f"### 🔢 Probabilité estimée : **{prob*100:.2f}%**")
        else:
            st.error("⚠️ Veuillez d'abord entraîner le modèle avant de faire une prédiction.")
