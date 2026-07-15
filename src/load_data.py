import pandas as pd

# Charger le dataset dans un DataFrame
df = pd.read_csv("data/telco_churn.csv")

# Regarder la "forme" : combien de lignes, combien de colonnes
print("Forme du dataset (lignes, colonnes) :", df.shape)

# Afficher les 5 premières lignes pour avoir un aperçu concret
print(df.head())

# Demander à pandas d'afficher TOUTES les colonnes (ne pas tronquer avec ...)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Inspecter les types de chaque colonne + les valeurs manquantes
print("\n--- Informations sur les colonnes ---")
print(df.info())

# Tenter de convertir TotalCharges en nombre.
# errors="coerce" : ce qui n'est PAS convertible devient NaN (valeur manquante),
# au lieu de planter. C'est une astuce clé pour débusquer les valeurs cachées.
total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Combien de valeurs sont devenues NaN ? = combien de cases n'étaient pas de vrais nombres
print("\n--- Diagnostic TotalCharges ---")
print("Nombre de valeurs non convertibles en nombre :", total_numeric.isna().sum())

# Regardons ces lignes problématiques directement
masque_problematique = total_numeric.isna()
print("\nLignes où TotalCharges n'est pas un nombre :")
print(df[masque_problematique][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])