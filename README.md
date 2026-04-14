# 🎓 Cours ADK — Agent IA connecté à une base de données

## ⏱ Durée : 15 minutes

---

## 🎬 L'histoire à raconter (pour le prof)

> **"Le data scientist qui parlait à sa base de données"**
>
> Imaginez : vous êtes data scientist dans une startup e-commerce tech.
> Chaque matin, le product manager débarque avec ses questions :
>
> *"C'est quoi notre best-seller ? On a des ruptures de stock ?
>  Y a eu des annulations récemment ?"*
>
> Vous ouvrez un notebook Jupyter, vous écrivez 3 requêtes SQL,
> vous formatez les résultats, vous les collez dans Slack…
>
> Un matin, vous vous dites :
>
> *"Et si le product manager pouvait poser ses questions en français
>  directement à la base de données ?"*
>
> Pas en lui donnant accès au SQL (danger !), mais en créant un **agent IA**
> qui sait quels outils utiliser pour interroger la base de façon **sécurisée**.
>
> C'est exactement ce qu'on va construire avec **Google ADK** et
> **MCP Toolbox for Databases** : un agent qui raisonne en langage naturel
> et exécute des requêtes SQL pré-définies.

---

## 🧱 Architecture de la démo

```
┌─────────────────────────┐
│  Agent ADK (Gemini)     │  ← Raisonne, choisit les outils
│  (sales_agent/agent.py) │
└────────┬────────────────┘
         │ HTTP (localhost:5000)
┌────────▼────────────────┐
│  MCP Toolbox Server     │  ← Expose les requêtes SQL comme des "outils"
│  (tools.yaml)           │     Pas de SQL injection possible
└────────┬────────────────┘
         │ SQLite
┌────────▼────────────────┐
│  shop.db                │  ← Base locale : 12 produits, 15 commandes
└─────────────────────────┘
```

**Pourquoi cette architecture ?**
- L'agent n'écrit **jamais** de SQL → pas de risque d'injection
- Les requêtes sont **pré-définies** dans `tools.yaml` → contrôle total
- L'agent choisit **quel outil** appeler selon la question → intelligent
- On peut changer la base (PostgreSQL, BigQuery…) **sans toucher** à l'agent

---

## 📚 Plan du cours

| Temps     | Étape                                                      |
|-----------|-----------------------------------------------------------|
| 0-3 min   | L'histoire + schéma d'architecture                         |
| 3-5 min   | Ouvrir `tools.yaml` : les outils SQL pré-définis          |
| 5-8 min   | Ouvrir `agent.py` : le ToolboxToolset et l'Agent          |
| 8-12 min  | Démo live : lancer `run_demo.py` (3 questions métier)     |
| 12-15 min | Bonus : `adk web` + questions du public                   |

---

## 🧠 Concepts clés

### Les 3 couches de sécurité

| Couche     | Rôle                                    |
|-----------|------------------------------------------|
| **Agent**  | Raisonne, ne voit QUE les descriptions  |
| **Toolbox**| Exécute des requêtes SQL paramétrées    |
| **SQLite** | Stocke les données                      |

L'agent ne connaît ni le schéma SQL, ni les tables, ni les colonnes.
Il voit uniquement les **noms** et **descriptions** des outils.

### Toolbox vs outils custom

| Outils à la main (`def func`)     | MCP Toolbox                        |
|-----------------------------------|------------------------------------|
| Vous écrivez du Python             | Vous écrivez du YAML              |
| Vous gérez la connexion DB         | Toolbox gère tout                 |
| Risque de SQL injection            | Requêtes paramétrées              |
| Couplé à votre code                | Découplé (microservice)           |
| OK pour du prototypage             | Production-ready                  |

---

## 🚀 Lancement rapide

```bash
# 1. Environnement virtuel
python3 -m venv .venv && source .venv/bin/activate

# 2. Dépendances
pip install -r requirements.txt

# 3. Clé API Gemini → https://aistudio.google.com/app/apikey
echo 'GOOGLE_API_KEY="votre-clé"' > sales_agent/.env

# 4. Créer la base de données
python setup_db.py

# 5. Lancer le serveur Toolbox (dans un terminal séparé)
./toolbox --config tools.yaml

# 6. Lancer la démo (dans un autre terminal)
python run_demo.py

# 7. (Bonus) Interface web interactive
adk web --port 8000
```

---

## 📁 Structure du projet

```
test_adk/
├── README.md          ← Ce fichier (notes du prof)
├── requirements.txt   ← Dépendances Python
├── setup_db.py        ← Script de création de la base SQLite
├── shop.db            ← Base de données (générée par setup_db.py)
├── tools.yaml         ← Configuration MCP Toolbox (les outils SQL)
├── toolbox            ← Binaire MCP Toolbox server
├── run_demo.py        ← Script de démo autonome
└── sales_agent/       ← Package agent ADK
    ├── __init__.py
    ├── .env           ← Clé API Gemini
    └── agent.py       ← ⭐ Le code de l'agent (15 lignes utiles !)
```

---

## 💡 Pour aller plus loin

- Remplacer SQLite par **PostgreSQL** ou **BigQuery** (changer juste le `source` dans tools.yaml)
- Ajouter des outils d'**écriture** (UPDATE, INSERT) pour réserver, modifier des commandes
- Combiner avec des **outils custom** Python (graphiques, exports CSV)
- **Multi-agents** : un agent "vendeur" + un agent "analyste" qui collaborent
- **Authentification** : restreindre les données par utilisateur connecté
- Docs : https://mcp-toolbox.dev/ et https://google.github.io/adk-docs/
