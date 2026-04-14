"""
🎓 Script de démo — Lancer l'agent Sales Analyst
=================================================

Ce script exécute l'agent ADK de façon programmatique
et affiche chaque étape (appels d'outils, résultats, réponse).
"""

import asyncio
import json
import logging
from dotenv import load_dotenv

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from sales_agent.agent import root_agent

load_dotenv("sales_agent/.env")

logging.basicConfig(
    level=logging.WARNING,
    format="   📋 %(name)s: %(message)s",
)

APP_NAME = "sales_demo"
USER_ID = "prof_demo"
SESSION_ID = "session_cours"


async def ask_agent(runner: Runner, session_id: str, question: str):
    """Envoie une question à l'agent et affiche sa réponse."""
    print(f"\n{'='*60}")
    print(f"👤 Question : {question}")
    print(f"{'='*60}")

    message = types.Content(
        role="user",
        parts=[types.Part(text=question)],
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=message
    ):
        function_calls = event.get_function_calls()
        if function_calls:
            for fc in function_calls:
                print(f"\n🔧 L'agent appelle l'outil : {fc.name}")
                print(f"   Paramètres : {dict(fc.args)}")

        function_responses = event.get_function_responses()
        if function_responses:
            for fr in function_responses:
                resp_data = fr.response if hasattr(fr, "response") else str(fr)
                preview = json.dumps(resp_data, ensure_ascii=False, indent=2)[:600]
                print(f"\n📦 Résultat de {fr.name} :")
                print(f"   {preview}")

        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text

    print(f"\n🤖 Réponse de l'agent :\n{response_text}")
    return response_text


async def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  🎓 Démo ADK — Agent Sales Analyst + SQLite + MCP Toolbox  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  L'agent a 6 outils SQL chargés depuis MCP Toolbox :       ║")
    print("║   1. search-products-by-category                           ║")
    print("║   2. search-products-by-name                               ║")
    print("║   3. get-top-products                                      ║")
    print("║   4. get-sales-by-product                                  ║")
    print("║   5. get-recent-orders                                     ║")
    print("║   6. get-low-stock-products                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── Scénario de démo : 3 questions métier ──

    # Question 1 : L'agent interroge les ventes
    await ask_agent(
        runner, SESSION_ID,
        "Donne-moi le classement des ventes par produit. Quel est notre best-seller ?"
    )

    # Question 2 : L'agent croise produits et stocks
    await ask_agent(
        runner, SESSION_ID,
        "Quels produits ont un stock inférieur à 50 unités ? On doit passer commande ?"
    )

    # Question 3 : L'agent fait une analyse métier
    await ask_agent(
        runner, SESSION_ID,
        "Montre-moi les 5 dernières commandes. Y a-t-il des annulations ?"
    )

    print("\n" + "=" * 60)
    print("✅ Démo terminée ! L'agent a interrogé SQLite via MCP Toolbox.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
