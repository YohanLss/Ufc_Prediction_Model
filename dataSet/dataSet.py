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
data = pd.read_csv("ufc-fighters-statistics.csv")
fights_df = pd.read_csv("ufc-master.csv")            


#nettoyage du deuxieme dataSet
fights_df.columns = fights_df.columns.str.lower()
fights_df['date'] = pd.to_datetime(fights_df['date'], errors='coerce')
fights_df = fights_df[['date', 'redfighter', 'bluefighter', 'winner']].copy()


#nettoyage de données

# Traitement de l'âge(convertir date_of_birth en en **objets datetime**)
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], errors='coerce')
data['age'] = (pd.to_datetime("today") - data['date_of_birth']).dt.days // 365

#suppression des colonnes inutiles
data.drop(columns=['date_of_birth', 'nickname'], inplace=True)


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


#analyse l'historique de combat entre deux combattants
def get_head_to_head(fighter_a, fighter_b, df=fights_df):
    fa = fighter_a.strip().lower()
    fb = fighter_b.strip().lower()

    # Filtrer les combats entre A et B
    fights = df[
        ((df['redfighter'].str.lower() == fa) & (df['bluefighter'].str.lower() == fb)) |
        ((df['redfighter'].str.lower() == fb) & (df['bluefighter'].str.lower() == fa))
    ].sort_values(by='date')

    #si aucun combat entre A et B

    if fights.empty:
        return {
            "has_fought_before": False,
            "total_fights": 0,
            "fighterA_wins": 0,
            "fighterB_wins": 0,
            "last_winner": None,
            "last_fight_date": None
        }

    def resolve_winner(row):
        if row['winner'] == 'Red':
            return row['redfighter'].strip().lower()
        elif row['winner'] == 'Blue':
            return row['bluefighter'].strip().lower()
        return None                 #concerne les Draws


    #compte les victoires de chaque
    fights['winner_resolved'] = fights.apply(resolve_winner, axis=1)
    fighterA_wins = (fights['winner_resolved'] == fa).sum()
    fighterB_wins = (fights['winner_resolved'] == fb).sum()

    #dernier combat
    last_fight = fights.iloc[-1]
    last_winner = resolve_winner(last_fight)

    return {
        "has_fought_before": True,
        "total_fights": len(fights),
        "fighterA_wins": fighterA_wins,
        "fighterB_wins": fighterB_wins,
        "last_winner": last_winner.title() if last_winner else "Unknown",
        "last_fight_date": last_fight['date'].strftime("%Y-%m-%d") if pd.notnull(last_fight['date']) else None
    }

def predict_duel(fighterA, fighterB, stats_df=data, model=model_pipeline, history_df=fights_df):
    fa, fb = fighterA.strip(), fighterB.strip()

    # Récupère les stats de chaque combattant
    rowA = stats_df[stats_df['name'].str.lower() == fa.lower()]
    rowB = stats_df[stats_df['name'].str.lower() == fb.lower()]
    if rowA.empty or rowB.empty:
        return {"error": "Un ou les deux combattants sont introuvables."}

    X_A = rowA.drop(columns=['wins'])
    X_B = rowB.drop(columns=['wins'])

    # Prédictions
    predA = model.predict(X_A)[0]
    predB = model.predict(X_B)[0]

    # Historique commun
    history = get_head_to_head(fa, fb, df=history_df)

    probable = fa if predA > predB else (fb if predB > predA else "Égalité")

    return {
        "fighterA": fa,
        "fighterB": fb,
        "predicted_wins_A": round(predA, 1),
        "predicted_wins_B": round(predB, 1),
        "probable_winner": probable,
        "head_to_head": history
    }

# Split
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model_pipeline.fit(X_train, y_train)

# Prédictions et évaluation

#y_pred = model_pipeline.predict(X_test)
#rmse = mean_squared_error(y_test, y_pred, squared=False)
#print(f"RMSE : {rmse:.2f}")

# Visualisation
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x=y_test, y=y_pred)
#plt.xlabel("Victoires réelles")
#plt.ylabel("Victoires prédites")
#plt.title("Comparaison des victoires réelles vs prédites")
#plt.grid(True)
#plt.show()
result = predict_duel("Ilia Topuria", "Leon Edwards")
print(result)