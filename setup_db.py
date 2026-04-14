"""
🎓 Script de setup — Créer et peupler la base SQLite
=====================================================

Génère une base de données réaliste avec :
  - 12 produits tech (avec prix, coût, stock, rating)
  - 200 clients avec profils variés
  - ~5000 commandes sur 3 ans (avril 2023 → avril 2026)

Les données incluent des patterns réalistes exploitables en ML :
  - Saisonnalité (pics à Noël, Black Friday, rentrée)
  - Clients récurrents vs ponctuels
  - Taux d'annulation variable selon le prix
  - Croissance progressive du volume de commandes
  - Corrélation prix/quantité (on commande plus d'accessoires pas chers)
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta
import math

DB_PATH = os.path.join(os.path.dirname(__file__), "shop.db")

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# DONNÉES DE RÉFÉRENCE
# ═══════════════════════════════════════════════════════════════════════

PRODUCTS = [
    #  id   name                        category        price   cost   stock  rating
    (1,  "Casque Bluetooth Pro",      "Audio",         89.99,  35.00,  120, 4.5),
    (2,  "Souris Gaming RGB",         "Périphérique",   49.99,  18.50,  250, 4.2),
    (3,  "Clavier Mécanique TKL",     "Périphérique",  129.99,  52.00,   80, 4.7),
    (4,  "Webcam 4K HDR",             "Vidéo",         159.99,  72.00,   45, 4.3),
    (5,  "Hub USB-C 7 ports",         "Accessoire",     39.99,  12.50,  300, 4.1),
    (6,  "Écran 27\" 4K IPS",         "Écran",         449.99, 280.00,   30, 4.8),
    (7,  "SSD NVMe 1To",              "Stockage",       79.99,  38.00,  200, 4.6),
    (8,  "Batterie externe 20000mAh", "Accessoire",     29.99,   9.50,  500, 4.0),
    (9,  "Micro USB condensateur",    "Audio",          69.99,  22.00,   90, 4.4),
    (10, "Support laptop ergonomique","Accessoire",     34.99,  14.00,  180, 4.3),
    (11, "Câble HDMI 2.1 3m",         "Accessoire",     19.99,   4.50,  600, 4.1),
    (12, "Enceinte portable",          "Audio",          59.99,  25.00,  150, 4.2),
]

FIRST_NAMES = [
    "Alice", "Bob", "Claire", "David", "Emma", "Frank", "Grace", "Hugo",
    "Iris", "Jules", "Karen", "Léo", "Marie", "Nicolas", "Olivia", "Paul",
    "Quentin", "Rose", "Sophie", "Thomas", "Ugo", "Valérie", "William",
    "Xavier", "Yasmine", "Zoé", "Adrien", "Béatrice", "Cédric", "Diane",
    "Élodie", "Fabien", "Gaëlle", "Henri", "Inès", "Jean", "Karine",
    "Laurent", "Manon", "Nathan", "Ophélie", "Pierre", "Raphaël", "Sarah",
    "Théo", "Ursule", "Vincent", "Wendy", "Yann", "Aurélie",
]

LAST_NAMES = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit",
    "Durand", "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel",
    "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier",
    "Morel", "Girard", "André", "Mercier", "Blanc", "Guérin", "Boyer",
    "Garnier", "Chevalier", "François", "Legrand", "Gauthier", "Perrin",
    "Robin", "Clément", "Morin", "Nicolas", "Henry", "Rousseau", "Mathieu",
]

CITIES = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Lille",
    "Nantes", "Strasbourg", "Rennes", "Montpellier", "Nice", "Grenoble",
    "Rouen", "Dijon", "Clermont-Ferrand", "Tours",
]


def generate_customers(n=200):
    """Génère n clients uniques avec ville et date d'inscription."""
    customers = []
    used_names = set()

    while len(customers) < n:
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        if name in used_names:
            continue
        used_names.add(name)

        city = random.choice(CITIES)
        # Date d'inscription entre avril 2023 et mars 2026
        days_offset = random.randint(0, 1050)
        signup_date = datetime(2023, 4, 1) + timedelta(days=days_offset)

        customers.append((len(customers) + 1, name, city, signup_date.strftime("%Y-%m-%d")))

    return customers


def seasonal_multiplier(date: datetime) -> float:
    """Retourne un multiplicateur saisonnier réaliste pour le e-commerce."""
    month = date.month
    day = date.day

    # Black Friday (fin novembre)
    if month == 11 and day >= 20:
        return 3.0
    # Noël (décembre)
    if month == 12:
        return 2.5 if day <= 20 else 1.0
    # Soldes d'hiver (janvier)
    if month == 1 and day <= 15:
        return 1.8
    # Rentrée (septembre)
    if month == 9:
        return 1.5
    # Soldes d'été (fin juin - juillet)
    if month == 6 and day >= 20:
        return 1.4
    if month == 7:
        return 1.3
    # Creux d'été (août)
    if month == 8:
        return 0.6
    # Creux de février
    if month == 2:
        return 0.7

    return 1.0


def growth_multiplier(date: datetime) -> float:
    """Croissance progressive de la startup sur 3 ans."""
    start = datetime(2023, 4, 1)
    months_elapsed = (date.year - start.year) * 12 + (date.month - start.month)
    # Croissance de 1.0 à ~2.5 sur 36 mois (courbe logistique)
    return 1.0 + 1.5 / (1 + math.exp(-0.1 * (months_elapsed - 18)))


def generate_orders(products, customers):
    """Génère des commandes réalistes sur 3 ans."""
    orders = []

    product_prices = {p[0]: p[3] for p in products}
    # Les produits pas chers sont commandés plus souvent
    product_weights = {p[0]: max(1, 10 - int(p[3] / 50)) for p in products}
    product_ids = list(product_weights.keys())
    weights = list(product_weights.values())

    # Certains clients sont des acheteurs réguliers (top 20%)
    n_regulars = len(customers) // 5
    regular_ids = [c[0] for c in customers[:n_regulars]]
    occasional_ids = [c[0] for c in customers[n_regulars:]]

    customer_signup = {c[0]: datetime.strptime(c[3], "%Y-%m-%d") for c in customers}

    start_date = datetime(2023, 4, 1)
    end_date = datetime(2026, 4, 13)
    current = start_date

    order_id = 1

    while current <= end_date:
        base_orders_per_day = 4.0
        seasonal = seasonal_multiplier(current)
        growth = growth_multiplier(current)

        # Moins de commandes le dimanche
        weekday_factor = 0.4 if current.weekday() == 6 else (0.8 if current.weekday() == 5 else 1.0)

        expected = base_orders_per_day * seasonal * growth * weekday_factor
        n_orders = max(0, int(random.gauss(expected, expected * 0.3)))

        for _ in range(n_orders):
            # 70% de chance d'être un client régulier
            if random.random() < 0.7 and regular_ids:
                cust_id = random.choice(regular_ids)
            else:
                cust_id = random.choice(occasional_ids)

            # Le client ne peut commander qu'après son inscription
            if current < customer_signup[cust_id]:
                cust_id = random.choice([c for c in regular_ids if customer_signup[c] <= current] or [customers[0][0]])

            product_id = random.choices(product_ids, weights=weights, k=1)[0]
            price = product_prices[product_id]

            # Quantité : inversement proportionnelle au prix
            if price < 30:
                qty = random.choices([1, 2, 3, 5, 10], weights=[30, 30, 20, 15, 5], k=1)[0]
            elif price < 80:
                qty = random.choices([1, 2, 3], weights=[50, 35, 15], k=1)[0]
            elif price < 200:
                qty = random.choices([1, 2], weights=[80, 20], k=1)[0]
            else:
                qty = 1

            total = round(price * qty, 2)

            # Statut : taux d'annulation corrélé au prix
            annul_rate = 0.02 + (price / 500) * 0.08  # 2% pour 0€, ~10% pour 450€
            if current > end_date - timedelta(days=7):
                status = random.choices(["livrée", "en cours"], weights=[30, 70], k=1)[0]
            elif random.random() < annul_rate:
                status = "annulée"
            elif current > end_date - timedelta(days=30):
                status = random.choices(["livrée", "en cours"], weights=[70, 30], k=1)[0]
            else:
                status = "livrée"

            # Heure de commande (distribution réaliste)
            hour = int(random.gauss(14, 4))
            hour = max(6, min(23, hour))
            minute = random.randint(0, 59)
            order_datetime = current.replace(hour=hour, minute=minute)

            orders.append((
                order_id,
                product_id,
                qty,
                total,
                customers[cust_id - 1][1],  # nom du client
                order_datetime.strftime("%Y-%m-%d %H:%M"),
                status,
            ))
            order_id += 1

        current += timedelta(days=1)

    return orders


def create_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"♻️  Ancienne base supprimée")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── Table produits ──
    cursor.execute("""
        CREATE TABLE products (
            id          INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            price       REAL    NOT NULL,
            cost        REAL    NOT NULL,
            stock       INTEGER NOT NULL,
            rating      REAL    NOT NULL
        )
    """)
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", PRODUCTS
    )

    # ── Table clients ──
    cursor.execute("""
        CREATE TABLE customers (
            id          INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            city        TEXT    NOT NULL,
            signup_date TEXT    NOT NULL
        )
    """)
    customers = generate_customers(200)
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)", customers
    )

    # ── Table commandes ──
    cursor.execute("""
        CREATE TABLE orders (
            id           INTEGER PRIMARY KEY,
            product_id   INTEGER NOT NULL,
            quantity     INTEGER NOT NULL,
            total_price  REAL    NOT NULL,
            customer     TEXT    NOT NULL,
            order_date   TEXT    NOT NULL,
            status       TEXT    NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    orders = generate_orders(PRODUCTS, customers)
    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders
    )

    conn.commit()

    # ── Stats ──
    cursor.execute("SELECT COUNT(*) FROM orders")
    n_orders = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM customers")
    n_customers = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
    date_min, date_max = cursor.fetchone()
    cursor.execute("SELECT ROUND(SUM(total_price), 2) FROM orders WHERE status != 'annulée'")
    ca_total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'annulée'")
    n_cancelled = cursor.fetchone()[0]
    cursor.execute("""
        SELECT strftime('%Y', order_date) as year, COUNT(*), ROUND(SUM(total_price), 2)
        FROM orders WHERE status != 'annulée'
        GROUP BY year ORDER BY year
    """)
    yearly = cursor.fetchall()

    conn.close()

    print(f"✅ Base créée : {DB_PATH}")
    print(f"   → {len(PRODUCTS)} produits")
    print(f"   → {n_customers} clients")
    print(f"   → {n_orders} commandes ({date_min} → {date_max})")
    print(f"   → {n_cancelled} annulations ({n_cancelled/n_orders*100:.1f}%)")
    print(f"   → CA total (hors annulations) : {ca_total:,.2f} €")
    print(f"\n   📊 Répartition par année :")
    for year, count, ca in yearly:
        print(f"      {year} : {count:>5} commandes — {ca:>12,.2f} € CA")

    print(f"\n   🧪 Patterns ML inclus :")
    print(f"      - Saisonnalité (Black Friday, Noël, soldes, rentrée)")
    print(f"      - Croissance startup (volume ×2.5 sur 3 ans)")
    print(f"      - Clients réguliers vs ponctuels (20/80)")
    print(f"      - Taux d'annulation corrélé au prix")
    print(f"      - Quantité inversement proportionnelle au prix")


if __name__ == "__main__":
    create_database()
