from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import preparer_donnees, construire_preprocesseur


def entrainer_modele(chemin_csv="data/telco_churn.csv"):
    """
    Assemble un Pipeline (préparation + régression logistique),
    l'entraîne sur le train, et retourne le modèle + les données de test.
    """
    # Récupérer le split stratifié préparé dans features.py
    X_train, X_test, y_train, y_test = preparer_donnees(chemin_csv)

    # Construire le préprocesseur (scaling + one-hot) sur la base du train
    preprocesseur = construire_preprocesseur(X_train)

    # Assembler le Pipeline : préparation PUIS modèle, en un seul objet.
    modele = Pipeline(steps=[
        ("preprocesseur", preprocesseur),
        ("classifieur", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # Entraîner : apprend les transformations ET le modèle, sur le train uniquement.
    modele.fit(X_train, y_train)

    return modele, X_test, y_test

if __name__ == "__main__":
    modele, X_test, y_test = entrainer_modele()
    print("\nModèle entraîné avec succès !")

    # Prédire sur le test (données jamais vues à l'entraînement)
    predictions = modele.predict(X_test)

    # Score brut (accuracy) — à interpréter avec prudence, on en a parlé
    score = modele.score(X_test, y_test)
    print("Accuracy sur le test :", round(score * 100, 1), "%")

    # Regarder les 10 premières prédictions vs la réalité
    print("\n10 premières prédictions :", predictions[:10])
    print("10 vraies valeurs        :", y_test.values[:10])