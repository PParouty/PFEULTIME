#!/usr/bin/env python3
"""
SIREN Investigation Agent - CLI
Système multi-agents pour l'exploration de graphes d'investigation.

Usage:
    python main.py                    # Mode interactif
    python main.py "votre question"   # Question unique
"""

import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def check_configuration():
    """Vérifie que la configuration est correcte."""
    errors = []

    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY non définie. Créez un fichier .env avec votre clé.")

    if not os.getenv("ES_URL"):
        print("⚠️  ES_URL non définie, utilisation de https://localhost:9220 par défaut")

    if errors:
        print("\n❌ Erreurs de configuration:")
        for error in errors:
            print(f"   - {error}")
        print("\nConsultez .env.example pour voir les variables requises.")
        sys.exit(1)


def print_banner():
    """Affiche la bannière du programme."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           SIREN Investigation Agent                           ║
║           Système Multi-Agents avec LangGraph                 ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_mode(orchestrator):
    """Mode interactif - conversation continue."""
    print("Mode interactif. Tapez 'quit' ou 'exit' pour quitter.\n")

    while True:
        try:
            query = input("\n🔍 Votre question: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\nAu revoir!")
                break

            if query.lower() == "help":
                print_help()
                continue

            # Exécuter la requête
            result = orchestrator.run(query)

            print("\n" + "="*60)
            print("📋 RÉSULTAT")
            print("="*60)
            print(result)

        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Réessayez ou tapez 'quit' pour quitter.")


def single_query_mode(orchestrator, query: str):
    """Mode requête unique."""
    result = orchestrator.run(query)

    print("\n" + "="*60)
    print("📋 RÉSULTAT")
    print("="*60)
    print(result)


def print_help():
    """Affiche l'aide."""
    help_text = """
📖 AIDE - SIREN Investigation Agent

Exemples de questions que vous pouvez poser:
─────────────────────────────────────────────────────────────
• "Trouve un lien entre l'entreprise MongoDB et 10gen"
• "Quels sont les investisseurs de l'entreprise Uber?"
• "Trouve les investissements en USD entre 2010 et 2015"
• "Quelles entreprises ont des investisseurs communs avec Airbnb?"

Commandes:
─────────────────────────────────────────────────────────────
• help  - Affiche cette aide
• quit  - Quitte le programme
• exit  - Quitte le programme

Configuration (.env):
─────────────────────────────────────────────────────────────
• OPENAI_API_KEY  - Votre clé API OpenAI (obligatoire)
• OPENAI_MODEL    - Modèle à utiliser (défaut: gpt-4o-mini)
• ES_URL          - URL Elasticsearch (défaut: https://localhost:9220)
• ES_USER         - Utilisateur Elasticsearch
• ES_PASSWORD     - Mot de passe Elasticsearch
    """
    print(help_text)


def main():
    """Point d'entrée principal."""
    print_banner()

    # Vérifier la configuration
    check_configuration()

    print("Initialisation du système...")

    # Importer ici pour éviter les erreurs si la config manque
    from core.orchestrator import GraphOrchestrator

    # Créer l'orchestrateur
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    orchestrator = GraphOrchestrator(model_name=model)

    print(f"✅ Système initialisé (modèle: {model})")

    # Déterminer le mode
    if len(sys.argv) > 1:
        # Mode requête unique
        query = " ".join(sys.argv[1:])
        single_query_mode(orchestrator, query)
    else:
        # Mode interactif
        interactive_mode(orchestrator)


if __name__ == "__main__":
    main()
