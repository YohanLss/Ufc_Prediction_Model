import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Chargement
data = pd.read_csv("dataSet/ufc-fighters-statistics.csv")


#nettoyage de données

# Traitement de l'âge(convertir date_of_birth en en **objets datetime**)
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], errors='coerce')
data['age'] = (pd.to_datetime("today") - data['date_of_birth']).dt.days // 365

#suppression des colonnes inutiles
data.drop(columns=['date_of_birth', 'name', 'nickname'], inplace=True)

# Cible et attributs
y = data['wins']                             #variable à prédire
X = data.drop(columns=['wins'])              #toutes les colonnes en elevant la variable "wins"

# Détection des colonnes
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = ['stance']

# Colonnes à transformer avec log(1+x)
log_transform_cols = [
    'losses', 'draws', 'significant_strikes_landed_per_minute',
    'significant_strikes_absorbed_per_minute', 'average_takedowns_landed_per_15_minutes'
]

# Pipeline numérique
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(
        lambda X: np.log1p(X) if isinstance(X, np.ndarray) else X,
        validate=False
    ))
])

# Pipeline catégoriel
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Prétraitement global
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Pipeline complet
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Split
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model_pipeline.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model_pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE : {rmse:.2f}")

# Visualisation
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Victoires réelles")
plt.ylabel("Victoires prédites")
plt.title("Comparaison des victoires réelles vs prédites")
plt.grid(True)
plt.show()
