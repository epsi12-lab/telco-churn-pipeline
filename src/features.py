import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from clean import load_and_clean


def preparer_donnees(chemin_csv="data/telco_churn.csv", test_size=0.2, random_state=42):
    """
    Charge les données propres, sépare X (variables) et y (cible),
    puis coupe en train/test de façon stratifiée.
    """
    df = load_and_clean(chemin_csv)

    # Convertir la cible Churn (Yes/No) en 1/0 : le modèle veut un nombre.
    y = (df["Churn"] == "Yes").astype(int)

    # X = toutes les colonnes SAUF la cible.
    X = df.drop("Churn", axis=1)

    # Couper en train (80%) et test (20%).
    # stratify=y : garde la même proportion 73/27 de churn dans les deux paquets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def construire_preprocesseur(X):
    """
    Construit le ColumnTransformer qui prépare les colonnes :
    - StandardScaler sur les colonnes numériques (mise à l'échelle)
    - OneHotEncoder sur les colonnes catégorielles (texte -> colonnes binaires)
    """
    # Détecter automatiquement les colonnes numériques et catégorielles.
    colonnes_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    colonnes_cat = X.select_dtypes(include=["object", "str"]).columns.tolist()

    print("Colonnes numériques :", colonnes_num)
    print("Colonnes catégorielles :", len(colonnes_cat), "colonnes")

    # Le ColumnTransformer applique le bon traitement à chaque groupe de colonnes.
    preprocesseur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_num),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), colonnes_cat),
        ]
    )

    return preprocesseur

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preparer_donnees()
    print("Taille train :", X_train.shape)
    print("Taille test  :", X_test.shape)
    print("Proportion churn train :", round(y_train.mean() * 100, 1), "%")
    print("Proportion churn test  :", round(y_test.mean() * 100, 1), "%\n")

    preprocesseur = construire_preprocesseur(X_train)
    # On "apprend" les transformations sur le train, puis on transforme le train.
    X_train_prepare = preprocesseur.fit_transform(X_train)
    print("\nForme du train APRÈS préparation :", X_train_prepare.shape)