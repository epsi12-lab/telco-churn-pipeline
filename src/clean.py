import pandas as pd


def load_and_clean(chemin_csv="data/telco_churn.csv"):
    """
    Charge le dataset Telco et applique le nettoyage :
    - conversion de TotalCharges en nombre
    - suppression des 11 lignes sans historique de facturation (tenure=0)
    - suppression de la colonne identifiant customerID
    Retourne un DataFrame propre, prêt pour l'analyse.
    """
    # Charger les données brutes
    df = pd.read_csv(chemin_csv)

    # Convertir TotalCharges (texte) en nombre.
    # errors="coerce" transforme les cases non numériques (les 11 espaces vides) en NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Supprimer les 11 lignes où TotalCharges est NaN (clients tenure=0, sans historique).
    # dropna() retire les lignes contenant des valeurs manquantes.
    # subset=["TotalCharges"] : on ne regarde QUE cette colonne pour décider.
    df = df.dropna(subset=["TotalCharges"])

    # Supprimer customerID : identifiant unique, aucune valeur prédictive.
    # axis=1 signifie "colonne" (axis=0 serait une ligne).
    df = df.drop("customerID", axis=1)
    
    return df

# Ce bloc ne s'exécute QUE si on lance directement "python src/clean.py".
# Il sert à tester/vérifier notre fonction.
if __name__ == "__main__":
    df = load_and_clean()
    print("Forme après nettoyage :", df.shape)
    print("\nTypes des colonnes numériques clés :")
    print(df[["tenure", "MonthlyCharges", "TotalCharges"]].dtypes)
    print("\nValeurs manquantes restantes par colonne :")
    print(df.isna().sum().sum(), "valeur(s) manquante(s) au total")