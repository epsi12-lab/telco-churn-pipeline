import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from features import preparer_donnees, construire_preprocesseur


def comparer_modeles(chemin_csv="data/telco_churn.csv"):
    """
    Entraîne deux modèles (régression logistique vs Random Forest)
    sur le même split, et compare leur AUC.
    """
    X_train, X_test, y_train, y_test = preparer_donnees(chemin_csv)

    # Un dictionnaire des deux modèles à comparer
    modeles = {
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    resultats = {}  # pour stocker les infos de chaque modèle

    for nom, classifieur in modeles.items():
        # Chaque modèle a SON propre préprocesseur, dans son propre Pipeline
        preprocesseur = construire_preprocesseur(X_train)
        pipe = Pipeline(steps=[
            ("preprocesseur", preprocesseur),
            ("classifieur", classifieur),
        ])
        pipe.fit(X_train, y_train)

        # Probabilités de churn pour calculer l'AUC
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        resultats[nom] = {"auc": auc, "proba": y_proba}

        print(f"\n=== {nom} ===")
        print(f"AUC : {auc:.3f}")
        print(classification_report(y_test, pipe.predict(X_test),
                                    target_names=["Reste", "Churn"]))

    # --- Courbe ROC comparative des deux modèles ---
    plt.figure(figsize=(7, 6))
    couleurs = {"Régression logistique": "#e74c3c", "Random Forest": "#2980b9"}

    for nom, infos in resultats.items():
        fpr, tpr, _ = roc_curve(y_test, infos["proba"])
        plt.plot(fpr, tpr, color=couleurs[nom], lw=2,
                 label=f"{nom} (AUC = {infos['auc']:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Hasard")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs (rappel)")
    plt.title("Comparaison ROC : Régression logistique vs Random Forest")
    plt.legend(loc="lower right")
    plt.savefig("reports/figures/roc_comparaison.png", dpi=150, bbox_inches="tight")
    print("\nCourbe comparative sauvegardée dans reports/figures/roc_comparaison.png")

    return resultats, y_test


if __name__ == "__main__":
    comparer_modeles()