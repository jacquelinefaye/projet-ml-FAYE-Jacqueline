"""Entraînement : compare plusieurs modèles, garde le meilleur, l'enregistre."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flux_train.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]
TARGET = "attaque"


def build_pipeline(model) -> Pipeline:
    """Prétraitement + modèle dans un seul Pipeline (rien à refaire au moment de predict)."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = pd.read_csv(DATA)
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Répartition de la cible :\n", df[TARGET].value_counts(normalize=True).round(3))

    X, y = df[NUM + CAT], df[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    candidats = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=42, n_jobs=-1),
    }

    scores = {}
    for nom, model in candidats.items():
        pipe = build_pipeline(model).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        f1 = f1_score(y_val, pred)
        scores[nom] = f1
        print(f"\n===== {nom} =====")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print("Matrice de confusion [[VN FP] [FN VP]] :\n", confusion_matrix(y_val, pred))

    meilleur = max(scores, key=scores.get)
    print(f"\nMeilleur modèle : {meilleur} (F1 attaque = {scores[meilleur]:.3f})")

    # Ré-entraînement sur TOUTES les données puis sauvegarde
    final = build_pipeline(candidats[meilleur]).fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print("Modèle enregistré dans", MODEL_PATH)


if __name__ == "__main__":
    main()
