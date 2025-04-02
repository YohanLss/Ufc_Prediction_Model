# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Chargement des données
# Assurez-vous d'avoir un fichier CSV avec des colonnes comme "combat_id", "combattant1", "combattant2",
# "statistiques_combattant1", "statistiques_combattant2", "résultat", etc.
data = pd.read_csv('ufc_fights.csv')

# Exploration des données (optionnel)
print(data.head())

# Préparation des données
# On sélectionne les colonnes importantes : ici, on suppose que les colonnes 'stat_combattant1' et 'stat_combattant2' contiennent des statistiques
# et que la colonne 'résultat' contient 1 si le combattant 1 gagne et 0 sinon.
features = data[['stat_combattant1', 'stat_combattant2', 'taille_combattant1', 'taille_combattant2']]
target = data['résultat']

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialisation du modèle de forêts aléatoires
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle:", accuracy)
print("Rapport de classification:")
print(classification_report(y_test, y_pred))

# Exemple de prédiction pour un nouveau combat
nouveau_combat = np.array([[100, 90, 180, 175]])  # Exemples de statistiques pour les combattants 1 et 2
pred = model.predict(nouveau_combat)
print("Prédiction pour le nouveau combat:", "Combattant 1 gagne" if pred[0] == 1 else "Combattant 2 gagne")
