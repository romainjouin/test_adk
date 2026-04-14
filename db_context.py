"""
Contexte dynamique de la base de données.

Lu une seule fois au démarrage (import), puis injecté dans les instructions
des agents pour éviter les hallucinations sans rien coder en dur.
"""

import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "shop.db")


def _load_context() -> dict:
    """Lit les métadonnées clés de shop.db au démarrage."""
    if not os.path.exists(DB_PATH):
        return {"products_list": "(base non trouvée)", "stats_summary": ""}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name, category, price FROM products ORDER BY id")
    products = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM customers")
    n_customers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM orders")
    n_orders = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
    date_min, date_max = cursor.fetchone()

    cursor.execute("SELECT DISTINCT category FROM products ORDER BY category")
    categories = [r[0] for r in cursor.fetchall()]

    conn.close()

    products_list = "\n".join(
        f"- {name} ({cat}, {price:.2f} €)" for name, cat, price in products
    )

    stats_summary = (
        f"- {len(products)} produits, {n_customers} clients, "
        f"~{n_orders} commandes ({date_min[:10]} → {date_max[:10]})\n"
        f"- Catégories : {', '.join(categories)}"
    )

    return {
        "products_list": products_list,
        "stats_summary": stats_summary,
    }


_ctx = _load_context()

PRODUCTS_LIST: str = _ctx["products_list"]
STATS_SUMMARY: str = _ctx["stats_summary"]
TODAY: str = date.today().isoformat()
