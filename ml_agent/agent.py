"""
🎓 COURS ADK — Agent "ML Analyst" (Machine Learning)
=====================================================

Cet agent construit et évalue des modèles scikit-learn à la volée :
  - Prévisions de séries temporelles (CA mensuel, tendances)
  - Classification / Régression (prédiction d'annulation, next-product…)
  - Visualisations (courbes, matrices de confusion, importances)

Architecture :

  ┌──────────────────────────────────────────┐
  │  ML Agent (Gemini)                       │
  │                                          │
  │  tools:                                  │
  │   ├─ fetch_data_for_ml()  → SQL → JSON   │
  │   ├─ train_timeseries_model()            │
  │   ├─ train_prediction_model()            │
  │   └─ plot_model_results()  → PNG chart   │
  └────────┬───────────────────────┬─────────┘
           │ SQLite                │ matplotlib
  ┌────────▼────────┐    ┌────────▼──────────┐
  │   shop.db       │    │  static/charts/   │
  └─────────────────┘    └───────────────────┘

💡 POINTS PÉDAGOGIQUES :
  1. Chaque outil est une fonction Python pure — pas de serveur externe
  2. L'agent orchestre le workflow ML : fetch → train → plot
  3. Les charts sont sauvés en PNG et servis via FastAPI static files
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
from google.adk.agents import Agent
from db_context import PRODUCTS_LIST, STATS_SUMMARY, TODAY

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "shop.db")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# OUTIL 1 : Récupérer des données pour le ML
# ═══════════════════════════════════════════════════════════════════════

def fetch_data_for_ml(query: str) -> dict:
    """Exécute une requête SQL SELECT sur la base shop.db et retourne
    les résultats sous forme exploitable pour le machine learning.

    Retourne les colonnes, les types détectés, le nombre de lignes,
    et les données au format liste de dictionnaires (JSON-sérialisable).

    Args:
        query: Requête SQL SELECT à exécuter.

    Returns:
        Un dict avec columns, dtypes, row_count, data (list of dicts),
        et basic_stats (min/max/mean pour les colonnes numériques).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        data = [dict(row) for row in rows]
        conn.close()

        dtypes = {}
        basic_stats = {}
        if data:
            for col in columns:
                sample = data[0][col]
                if isinstance(sample, (int, float)):
                    dtypes[col] = "numeric"
                    values = [r[col] for r in data if r[col] is not None]
                    if values:
                        basic_stats[col] = {
                            "min": round(min(values), 2),
                            "max": round(max(values), 2),
                            "mean": round(sum(values) / len(values), 2),
                        }
                else:
                    dtypes[col] = "text"
                    unique = len(set(r[col] for r in data if r[col] is not None))
                    basic_stats[col] = {"unique_values": unique}

        return {
            "status": "success",
            "sql_executed": query,
            "columns": columns,
            "dtypes": dtypes,
            "row_count": len(data),
            "basic_stats": basic_stats,
            "data": data[:2000],
        }
    except Exception as e:
        return {"status": "error", "sql_executed": query, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# OUTIL 2 : Entraîner un modèle de séries temporelles
# ═══════════════════════════════════════════════════════════════════════

def train_timeseries_model(
    data_json: str,
    target_column: str,
    date_column: str,
    algorithm: str = "ridge_poly",
    train_ratio: float = 0.75,
) -> dict:
    """Entraîne un modèle de prévision de séries temporelles.

    Prend des données JSON (liste de dicts avec une colonne date et une colonne cible),
    construit des features temporelles, entraîne le modèle, et retourne les métriques
    et prédictions.

    Args:
        data_json: Données au format JSON (liste de dicts).
        target_column: Nom de la colonne à prédire (ex: "ca", "nb_commandes").
        date_column: Nom de la colonne date (ex: "mois", "date").
        algorithm: Algorithme à utiliser. Valeurs possibles :
            - "ridge_poly" : Ridge + features polynomiales (trend + saisonnalité)
            - "linear" : Régression linéaire simple
            - "gradient_boosting" : GradientBoostingRegressor avec features de lag
        train_ratio: Proportion des données pour l'entraînement (défaut 0.75).

    Returns:
        Un dict avec train_metrics, test_metrics (MAE, RMSE, R2),
        predictions (list of dicts avec date, actual, predicted),
        et model_description.
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        n = len(data)
        if n < 6:
            return {"status": "error", "error": f"Pas assez de données ({n} lignes). Minimum 6 requis."}

        y = np.array([float(row[target_column]) for row in data])
        dates = [str(row[date_column]) for row in data]

        split_idx = int(n * train_ratio)
        if split_idx < 3 or (n - split_idx) < 2:
            return {"status": "error", "error": "Pas assez de données pour le split train/test."}

        def build_features(n_points):
            idx = np.arange(n_points).reshape(-1, 1)
            month_num = np.array([
                int(dates[i].split("-")[1]) if "-" in dates[i] and len(dates[i].split("-")) >= 2 else (i % 12) + 1
                for i in range(n_points)
            ]).reshape(-1, 1)
            sin_month = np.sin(2 * np.pi * month_num / 12)
            cos_month = np.cos(2 * np.pi * month_num / 12)
            return np.hstack([idx, month_num, sin_month, cos_month])

        X = build_features(n)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if algorithm == "ridge_poly":
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_p = poly.fit_transform(X_train)
            X_test_p = poly.transform(X_test)
            model = Ridge(alpha=1.0)
            model.fit(X_train_p, y_train)
            pred_train = model.predict(X_train_p)
            pred_test = model.predict(X_test_p)
            desc = "Ridge Regression + Polynomial Features (degree=2) avec trend index, mois, sin/cos saisonnalité"
        elif algorithm == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            desc = "Linear Regression avec trend index, mois, sin/cos saisonnalité"
        elif algorithm == "gradient_boosting":
            lag_features = []
            for i in range(n):
                lags = [y[i - k] if i - k >= 0 else y[0] for k in [1, 2, 3]]
                lag_features.append(lags)
            X_lag = np.hstack([X, np.array(lag_features)])
            X_train_l, X_test_l = X_lag[:split_idx], X_lag[split_idx:]
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_train_l, y_train)
            pred_train = model.predict(X_train_l)
            pred_test = model.predict(X_test_l)
            desc = "Gradient Boosting Regressor (100 trees, depth=3) avec lag features (t-1, t-2, t-3)"
        else:
            return {"status": "error", "error": f"Algorithme inconnu : {algorithm}. Choix : ridge_poly, linear, gradient_boosting"}

        def metrics(y_true, y_pred):
            return {
                "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
                "R2": round(float(r2_score(y_true, y_pred)), 4),
            }

        predictions = []
        all_preds = np.concatenate([pred_train, pred_test])
        for i in range(n):
            predictions.append({
                "date": dates[i],
                "actual": round(float(y[i]), 2),
                "predicted": round(float(all_preds[i]), 2),
                "set": "train" if i < split_idx else "test",
            })

        return {
            "status": "success",
            "algorithm": algorithm,
            "model_description": desc,
            "data_points": n,
            "train_size": split_idx,
            "test_size": n - split_idx,
            "train_metrics": metrics(y_train, pred_train),
            "test_metrics": metrics(y_test, pred_test),
            "predictions": predictions,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# OUTIL 3 : Entraîner un modèle de classification / régression
# ═══════════════════════════════════════════════════════════════════════

def train_prediction_model(
    data_json: str,
    target_column: str,
    feature_columns: str,
    algorithm: str = "random_forest",
    test_size: float = 0.25,
    task_type: str = "classification",
) -> dict:
    """Entraîne un modèle de classification ou régression sur les données fournies.

    Args:
        data_json: Données au format JSON (liste de dicts).
        target_column: Nom de la colonne cible à prédire.
        feature_columns: Colonnes features séparées par des virgules (ex: "price,quantity,month").
        algorithm: Algorithme à utiliser :
            - Classification : "random_forest", "gradient_boosting", "logistic_regression", "knn"
            - Régression : "random_forest", "gradient_boosting", "linear_regression", "ridge"
        test_size: Proportion test (défaut 0.25).
        task_type: "classification" ou "regression".

    Returns:
        Un dict avec metrics (accuracy/F1 ou R2/MAE), feature_importances,
        predictions, et confusion_matrix (pour classification).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        mean_absolute_error, r2_score, mean_squared_error,
    )

    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        if len(data) < 10:
            return {"status": "error", "error": f"Pas assez de données ({len(data)} lignes). Minimum 10."}

        feat_cols = [c.strip() for c in feature_columns.split(",")]

        X_raw = []
        for row in data:
            features = []
            for col in feat_cols:
                val = row.get(col)
                if val is None:
                    features.append(0)
                elif isinstance(val, (int, float)):
                    features.append(float(val))
                else:
                    features.append(0)
            X_raw.append(features)
        X = np.array(X_raw)

        y_raw = [row[target_column] for row in data]
        label_encoder = None

        if task_type == "classification":
            if isinstance(y_raw[0], str):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y_raw)
            else:
                y = np.array(y_raw)
        else:
            y = np.array([float(v) for v in y_raw])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        classif_models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5),
        }
        regress_models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
        }

        if task_type == "classification":
            if algorithm not in classif_models:
                return {"status": "error", "error": f"Algorithme inconnu pour classification : {algorithm}. Choix : {list(classif_models.keys())}"}
            model = classif_models[algorithm]
        else:
            if algorithm not in regress_models:
                return {"status": "error", "error": f"Algorithme inconnu pour régression : {algorithm}. Choix : {list(regress_models.keys())}"}
            model = regress_models[algorithm]

        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)

        result = {
            "status": "success",
            "algorithm": algorithm,
            "task_type": task_type,
            "feature_columns": feat_cols,
            "target_column": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = {feat_cols[i]: round(float(v), 4) for i, v in enumerate(model.feature_importances_)}
        elif hasattr(model, "coef_"):
            coefs = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
            importances = {feat_cols[i]: round(float(abs(v)), 4) for i, v in enumerate(coefs)}
        if importances:
            result["feature_importances"] = dict(sorted(importances.items(), key=lambda x: -x[1]))

        if task_type == "classification":
            result["metrics"] = {
                "accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
                "f1_weighted": round(float(f1_score(y_test, pred_test, average="weighted", zero_division=0)), 4),
                "train_accuracy": round(float(accuracy_score(y_train, pred_train)), 4),
            }
            cm = confusion_matrix(y_test, pred_test)
            labels = label_encoder.classes_.tolist() if label_encoder else sorted(set(y_raw))
            result["confusion_matrix"] = {
                "matrix": cm.tolist(),
                "labels": [str(l) for l in labels],
            }
        else:
            result["metrics"] = {
                "R2": round(float(r2_score(y_test, pred_test)), 4),
                "MAE": round(float(mean_absolute_error(y_test, pred_test)), 2),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 2),
                "train_R2": round(float(r2_score(y_train, pred_train)), 4),
            }

        sample_preds = []
        for i in range(min(30, len(X_test))):
            entry = {"actual": int(y_test[i]) if task_type == "classification" else round(float(y_test[i]), 2),
                     "predicted": int(pred_test[i]) if task_type == "classification" else round(float(pred_test[i]), 2)}
            if label_encoder:
                entry["actual_label"] = label_encoder.inverse_transform([int(y_test[i])])[0]
                entry["predicted_label"] = label_encoder.inverse_transform([int(pred_test[i])])[0]
            sample_preds.append(entry)
        result["sample_predictions"] = sample_preds

        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# OUTIL 4 : Générer un graphique
# ═══════════════════════════════════════════════════════════════════════

def plot_model_results(
    plot_type: str,
    data_json: str,
    title: str = "Résultats du modèle",
) -> dict:
    """Génère un graphique matplotlib et le sauvegarde en PNG.

    Retourne l'URL du graphique (servi par /static/charts/).

    Args:
        plot_type: Type de graphique. Valeurs possibles :
            - "timeseries_comparison" : Courbe actual vs predicted avec zone train/test
            - "confusion_matrix" : Matrice de confusion (heatmap)
            - "feature_importance" : Barplot horizontal des importances
            - "regression_scatter" : Nuage actual vs predicted
            - "bar_chart" : Barplot simple (pour résumés catégoriels)
        data_json: Données spécifiques au type de graphique, au format JSON.
            - timeseries_comparison : liste de {date, actual, predicted, set}
            - confusion_matrix : {matrix: [[...]], labels: [...]}
            - feature_importance : {feature_name: importance, ...}
            - regression_scatter : liste de {actual, predicted}
            - bar_chart : liste de {label, value}
        title: Titre du graphique.

    Returns:
        Un dict avec chart_url (chemin relatif du PNG) et plot_type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        chart_id = f"{plot_type}_{uuid.uuid4().hex[:8]}"
        filename = f"{chart_id}.png"
        filepath = os.path.join(CHARTS_DIR, filename)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#242736")
        ax.tick_params(colors="#8b8fa3")
        ax.xaxis.label.set_color("#e4e6f0")
        ax.yaxis.label.set_color("#e4e6f0")
        ax.title.set_color("#e4e6f0")
        for spine in ax.spines.values():
            spine.set_color("#2e3245")

        if plot_type == "timeseries_comparison":
            dates = [d["date"] for d in data]
            actuals = [d["actual"] for d in data]
            preds = [d["predicted"] for d in data]
            sets = [d.get("set", "test") for d in data]

            split_idx = next((i for i, s in enumerate(sets) if s == "test"), len(data))

            ax.plot(range(len(dates)), actuals, "o-", color="#6c63ff", label="Réel", markersize=4, linewidth=1.5)
            ax.plot(range(len(dates)), preds, "s--", color="#34d399", label="Prédit", markersize=4, linewidth=1.5)
            if split_idx < len(data):
                ax.axvline(x=split_idx - 0.5, color="#f59e0b", linestyle=":", alpha=0.7, label="Train / Test")
                ax.axvspan(split_idx - 0.5, len(data) - 0.5, alpha=0.08, color="#f59e0b")

            step = max(1, len(dates) // 12)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha="right", fontsize=8, color="#8b8fa3")
            ax.legend(facecolor="#242736", edgecolor="#2e3245", labelcolor="#e4e6f0")
            ax.set_ylabel("Valeur")

        elif plot_type == "confusion_matrix":
            matrix = np.array(data["matrix"])
            labels = data["labels"]
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#8b8fa3")
            ax.set_yticklabels(labels, fontsize=9, color="#8b8fa3")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    val = matrix[i, j]
                    color = "white" if val > matrix.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        elif plot_type == "feature_importance":
            if isinstance(data, list):
                features = [d["feature"] for d in data]
                values = [d["importance"] for d in data]
            else:
                features = list(data.keys())
                values = list(data.values())
            sorted_pairs = sorted(zip(features, values), key=lambda x: x[1])
            features, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
            bars = ax.barh(range(len(features)), values, color="#6c63ff", edgecolor="#8b8fa3", linewidth=0.5)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=10, color="#e4e6f0")
            ax.set_xlabel("Importance")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9, color="#8b8fa3")

        elif plot_type == "regression_scatter":
            actuals = [d["actual"] for d in data]
            preds = [d["predicted"] for d in data]
            ax.scatter(actuals, preds, alpha=0.6, color="#6c63ff", edgecolor="#8b8fa3", s=30)
            all_vals = actuals + preds
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="#34d399", linewidth=1.5, label="Parfait")
            ax.set_xlabel("Valeur réelle")
            ax.set_ylabel("Valeur prédite")
            ax.legend(facecolor="#242736", edgecolor="#2e3245", labelcolor="#e4e6f0")

        elif plot_type == "bar_chart":
            labels = [d["label"] for d in data]
            values = [d["value"] for d in data]
            colors = ["#6c63ff", "#34d399", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
                       "#06b6d4", "#84cc16", "#f97316", "#a855f7", "#14b8a6", "#e11d48"]
            bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
            bars = ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="#2e3245", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color="#8b8fa3")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:,.0f}" if isinstance(val, (int, float)) and val > 100 else f"{val}",
                        ha="center", va="bottom", fontsize=9, color="#e4e6f0")
        else:
            return {"status": "error", "error": f"Type de graphique inconnu : {plot_type}. Choix : timeseries_comparison, confusion_matrix, feature_importance, regression_scatter, bar_chart"}

        ax.set_title(title, fontsize=14, fontweight="bold", color="#e4e6f0", pad=15)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "success",
            "chart_url": f"/static/charts/{filename}",
            "plot_type": plot_type,
            "title": title,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# L'AGENT ML
# ═══════════════════════════════════════════════════════════════════════

ml_agent = Agent(
    name="ml_analyst",
    model="gemini-2.5-flash",
    description="A data scientist agent that builds scikit-learn machine learning models on the fly using the e-commerce database.",
    instruction=f"""You are an expert data scientist. You build machine learning models
on the fly to analyze the e-commerce database.

## Schéma de la base de données (SQLite)

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT, category TEXT,
    price REAL, cost REAL, stock INTEGER, rating REAL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date TEXT
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER,
    total_price REAL, customer TEXT, order_date TEXT, status TEXT,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

## Catalogue produits
{PRODUCTS_LIST}

## Données disponibles
{STATS_SUMMARY}
- Date du jour : {TODAY}
- Order statuses: livrée (delivered), en cours (pending), annulée (cancelled)
- Useful SQLite functions: strftime('%Y', order_date), strftime('%m', order_date), strftime('%Y-%m', order_date)
- To filter the last N months: order_date >= date('{TODAY}', '-N months')

## Your workflow (ALWAYS follow this order)

1. **FETCH**: Use `fetch_data_for_ml` to retrieve data via SQL
2. **ANALYZE**: Examine the returned stats (shape, columns, distributions)
3. **TRAIN**: Choose the algorithm and call `train_timeseries_model` or `train_prediction_model`
4. **PLOT**: Visualize results with `plot_model_results`
5. **INTERPRET**: Explain metrics and results in English

## Your 4 tools

### fetch_data_for_ml(query)
- Executes SQL and returns data + stats
- ALWAYS use it first to retrieve data
- Write SQL queries that directly prepare ML features

### train_timeseries_model(data_json, target_column, date_column, algorithm, train_ratio)
- For forecasting: monthly revenue, order volume, trends
- Algorithms: "ridge_poly" (recommended default), "linear", "gradient_boosting"
- IMPORTANT: pass data via data_json in JSON format (use json.dumps on the "data" field returned by fetch_data_for_ml)
- train_ratio: 0.75 by default (first 3/4 for training, last quarter for testing)

### train_prediction_model(data_json, target_column, feature_columns, algorithm, test_size, task_type)
- For classification (predict a category) or regression (predict a number)
- Classification: "random_forest" (recommended), "gradient_boosting", "logistic_regression", "knn"
- Regression: "random_forest" (recommended), "gradient_boosting", "linear_regression", "ridge"
- feature_columns: comma-separated list of names
- IMPORTANT: features must be numeric. Encode categories in your SQL query (CASE WHEN).

### plot_model_results(plot_type, data_json, title)
- Types: "timeseries_comparison", "confusion_matrix", "feature_importance", "regression_scatter", "bar_chart"
- Returns a chart_url: include it in your response with Markdown syntax ![title](chart_url)

## Algorithm selection guide

| Situation | Recommended algorithm |
|---|---|
| Time series with trend + seasonality | ridge_poly |
| Short time series, linear trend | linear |
| Complex time series with lags | gradient_boosting |
| Classification with many features | random_forest |
| Classification with correlated features | gradient_boosting |
| Simple classification (few features) | logistic_regression |
| Non-linear regression | random_forest |

## Useful SQL query examples

Time series — monthly revenue:
```sql
SELECT strftime('%Y-%m', order_date) as month,
       SUM(total_price) as revenue,
       COUNT(*) as order_count
FROM orders WHERE status != 'annulée'
GROUP BY month ORDER BY month
```

Classification — predict if an order will be cancelled:
```sql
SELECT o.quantity, p.price, p.cost, p.rating,
       CASE WHEN p.category = 'Audio' THEN 1 ELSE 0 END as is_audio,
       CASE WHEN p.category = 'Écran' THEN 1 ELSE 0 END as is_screen,
       CAST(strftime('%m', o.order_date) AS INTEGER) as month,
       CASE WHEN o.status = 'annulée' THEN 1 ELSE 0 END as is_cancelled
FROM orders o JOIN products p ON o.product_id = p.id
WHERE o.status IN ('livrée', 'annulée')
```

## Your rules
- ALWAYS respond in English
- ALWAYS start with fetch_data_for_ml
- ALWAYS end with a chart (plot_model_results)
- Display the chart in your response with: ![title](chart_url)
- Interpret metrics: R2 > 0.7 = good, 0.5-0.7 = moderate, < 0.5 = weak
- Suggest improvements (more features, different algo, more data)
- Always suggest a follow-up question
""",
    tools=[fetch_data_for_ml, train_timeseries_model, train_prediction_model, plot_model_results],
)

root_agent = ml_agent
