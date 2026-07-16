import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from train import entrainer_modele


def evaluer_modele(chemin_csv="data/telco_churn.csv"):
    """
    Entraîne le modèle puis l'évalue en profondeur :
    matrice de confusion, précision, rappel, F1, ROC-AUC.
    """
    # Récupérer le modèle entraîné + les données de test
    modele, X_test, y_test = entrainer_modele(chemin_csv)

    # Prédictions sur le test
    y_pred = modele.predict(X_test)

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)

    # L'afficher joliment et la sauvegarder
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Reste (0)", "Churn (1)"])
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion — Régression logistique")
    plt.savefig("reports/figures/matrice_confusion.png", dpi=150, bbox_inches="tight")
    print("\nGraphique sauvegardé dans reports/figures/matrice_confusion.png")

    # --- Rapport de classification : précision, rappel, F1 ---
    from sklearn.metrics import classification_report
    print("\n--- Rapport de classification ---")
    print(classification_report(y_test, y_pred,
                                target_names=["Reste (0)", "Churn (1)"]))

    # --- Ajustement du seuil de décision ---
    import numpy as np
    from sklearn.metrics import classification_report as rapport

    # Récupérer les PROBABILITÉS de churn (colonne 1), pas les classes 0/1
    y_proba = modele.predict_proba(X_test)[:, 1]

    # Baisser le seuil de 0.50 à 0.35 : on déclenche l'alerte churn plus facilement
    seuil = 0.35
    y_pred_seuil = (y_proba >= seuil).astype(int)

    print(f"\n--- Rapport avec seuil abaissé à {seuil} ---")
    print(rapport(y_test, y_pred_seuil, target_names=["Reste (0)", "Churn (1)"]))

    # Matrice de confusion au seuil ajusté, sauvegardée pour le README
    cm_seuil = confusion_matrix(y_test, y_pred_seuil)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_seuil,
                                   display_labels=["Reste (0)", "Churn (1)"])
    disp2.plot(cmap="Oranges")
    plt.title(f"Matrice de confusion — seuil {seuil}")
    plt.savefig("reports/figures/matrice_confusion_seuil035.png", dpi=150, bbox_inches="tight")

    return modele, X_test, y_test, y_pred


if __name__ == "__main__":
    evaluer_modele()