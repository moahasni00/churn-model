
# 🛠️ Application de Prédiction du Churn – Programme de Fidélité Shell

Bienvenue dans cette application de **data science décisionnelle** développée avec **Streamlit**, qui permet de prédire le risque de départ (churn) des clients du programme de fidélité **Vivo Energy Maroc – Shell Licensee**, à partir de données simulées mais réalistes.

---

## 🚀 Démo en ligne

📍 [Clique ici pour accéder à l'application sur Streamlit Cloud](https://churn-modelfeg.streamlit.app/)

---

## 🎯 Objectif du projet

> Aider les décideurs marketing à **identifier les clients à risque** de quitter le programme de fidélité, en utilisant des modèles d’apprentissage automatique simples et interprétables.

L'application est un **outil d'aide à la décision** destiné aux responsables relation client, analystes marketing, et étudiants en data science.

---

## 🧪 Fonctionnalités clés

| Fonction | Description |
|----------|-------------|
| 📊 KPI Dashboard | Aperçu des clients, taux de churn, points moyens, ratio de fidélité |
| 🔍 Analyse Exploratoire | Graphiques interactifs (histogrammes, boxplots, heatmaps, etc.) |
| 🤖 Entraînement ML | Choix entre `Random Forest`, `Régression Logistique`, `KNN` |
| 🎯 Prédiction client | Simulation manuelle avec sliders + probabilité de churn |
| 📄 Rapport de performance | Score AUC, Accuracy, et classification report |

---

## 🗃️ Données utilisées

> Les données sont **synthétiques** (fictives), générées pour respecter l’éthique et la confidentialité tout en **reproduisant les échelles réelles** de Shell Maroc.

**Variables principales** :
- Points totaux & points utilisés
- Type de carburant utilisé
- Montant en lubrifiants / boutique / restaurant
- Fidélité estimée, ratio, etc.

[Client_ID,	Total_Points,	Fuel_Type,	Lubricant_Amount_MAD,	Shop_Amount_MAD,	Restaurant_Amount_MAD,	Butagaz_Units,	Points_Redeemed,	Estimated_Earned_Points,	Loyalty_Ratio,	Churned]


---

## 🧩 Technologies utilisées

- `Python 3.10`
- `Streamlit`
- `scikit-learn`
- `matplotlib` & `seaborn`
- `pandas`, `numpy`

---

## ▶️ Lancer l'application en local

### 1. Cloner le dépôt
```bash
git clone https://github.com/ton-utilisateur/ton-depot.git
cd ton-depot
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer Streamlit
```bash
streamlit run app.py
```

---

## 🖼️ Capture d’écran

![Screenshot](![image](https://github.com/user-attachments/assets/e049e584-7d41-47ef-84da-1ac8c112cb43)
![image](https://github.com/user-attachments/assets/d3bd2763-cb72-4e51-8119-4eba2b908e22)

---

## 👩‍🎓 Réalisée par
**Mohammed Amine Hasni**
Université Hassan I – Master en Ingénierie de la Décision
In corporation with :  
**Oumaima Zaz**  Université Hassan I – Master en Marketing et Action Commerciale 
© 2025 – Tous droits réservés

---

## 📝 Licence

Projet académique. Reproduction et réutilisation libre à des fins pédagogiques.
