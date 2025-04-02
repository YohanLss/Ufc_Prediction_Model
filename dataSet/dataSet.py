# Bibliothèques nécessaires
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charge les données
data = pd.read_csv("dataSet/ufc-fighters-statistics.csv")

# Vérifie les données et leurs types
print("Aperçu des données :")
print(data.head())
print("\nTypes de données :")
print(data.dtypes)

# Supprime les valeurs nulles (si nécessaire)
data.dropna(inplace=True)

# Définir les variables X (indépendantes) et Y (dépendantes)
x = data.drop(['wins'], axis=1)  # Supprimer la colonne 'wins' pour créer X
y = data['wins']  # Variable cible (à prédire)

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Joindre x_train et y_train pour l'analyse exploratoire
train_data = x_train.join(y_train)

# Conserve uniquement les colonnes numériques pour l'analyse
numeric_data = train_data.select_dtypes(include=[np.number])

# Vérifie les colonnes numériques sélectionnées
print("\nColonnes numériques dans les données d'entraînement :")
print(numeric_data.columns)

# Affiche des histogrammes pour visualiser la distribution des variables numériques
numeric_data.hist(figsize=(20, 10))
plt.suptitle("Distributions des variables numériques dans les données d'entraînement")
plt.show()

# Calculer et afficher la matrice de corrélation
plt.figure(figsize=(30, 40))
sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu")
plt.title("Matrice de corrélation des variables numériques")
plt.show()

#permet juste de normaliser les données tout en évitant
#les problèmes liés aux valeurs extrêmes
variables_to_transform = ['losses', 'draws', 'significant_strikes_landed_per_minute', 
                          'significant_strikes_absorbed_per_minute', 'average_takedowns_landed_per_15_minutes']

for var in variables_to_transform:
    train_data[var] = np.log(train_data[var] + 1) 

#juste pour afficher les histogrammes après la transformation
#for var in variables_to_transform:
    # plt.figure(figsize=(8, 4))
    # plt.hist(train_data[var], bins=30, alpha=0.7, color='orange')
    # plt.title(f"Distribution transformée de {var}")
    # plt.xlabel(f"Log({var})")
    # plt.ylabel("Fréquence")
    # plt.show()

#corr() ne prend en paramètre que des valeurs numériques , alors on convertit ces valeurs en valeurs numériques
# Vérifier si 'significant_strikes_absorbed_per_minute' est numérique ou catégorique
if train_data['significant_strikes_absorbed_per_minute'].dtype in [np.float64, np.int64]:
    # Regrouper en catégories avant d'utiliser get_dummies (si nécessaire)
    bins = [0, 1, 2, 3, np.inf]  # Modifier les seuils selon vos données
    labels = ['low', 'medium', 'high', 'very_high']
    train_data['significant_strikes_absorbed_per_minute_bins'] = pd.cut(
        train_data['significant_strikes_absorbed_per_minute'], bins=bins, labels=labels
    )

    # Convertir les catégories en colonnes binaires
    train_data = train_data.join(
        pd.get_dummies(train_data['significant_strikes_absorbed_per_minute_bins'], prefix='ssapm')
    ).drop(['significant_strikes_absorbed_per_minute', 'significant_strikes_absorbed_per_minute_bins'], axis=1)
else:
    # Utiliser directement get_dummies si la colonne est catégorique
    train_data = train_data.join(
        pd.get_dummies(train_data['significant_strikes_absorbed_per_minute'], prefix='ssapm')
    ).drop(['significant_strikes_absorbed_per_minute'], axis=1)

# Vérifier les types de colonnes
# print("\nTypes de colonnes dans train_data après transformation :")
# print(train_data.dtypes)

# Filtrer uniquement les colonnes numériques
numeric_data = train_data.select_dtypes(include=[np.number])

# Afficher la matrice de corrélation
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap="YlGnBu")
plt.title("Matrice de corrélation des variables numériques")
plt.show()

plt.figure(figsize=(15 , 8))
sns.scatterplot(x="wins" , y ="losses" , data = train_data, hue="draws", palette= "coolwarm")
plt.show()