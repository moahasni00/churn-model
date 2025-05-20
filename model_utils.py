import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# 🔄 Charger et préparer les données depuis l'URL GitHub
def load_and_prepare_data():
    url = "https://raw.githubusercontent.com/moahasni00/churnVivoEnergyData/refs/heads/main/shell_loyalty_churn_large.csv"
    df = pd.read_csv(url)
    
    # Encodage de la variable catégorielle
    df["Fuel_Type"] = LabelEncoder().fit_transform(df["Fuel_Type"])
    df.drop("Client_ID", axis=1, inplace=True)
    
    return df

# 🧠 Entraînement du modèle + retour des métriques
def train_model(df):
    X = df.drop("Churned", axis=1)
    y = df["Churned"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "classification": classification_report(y_test, y_pred, output_dict=True),
    }

    return model, scaler, metrics

# 📊 Matrice de confusion
def plot_confusion(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion")
    return fig
