# SIREN Investigation Agent

Système multi-agents orchestré par LLM pour l'exploration de graphes d'investigation SIREN.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATEUR (LangGraph)                │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Planificateur│  │ Priorisation│  │   Résumé    │         │
│  │    Agent    │  │    Agent    │  │    Agent    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │         Outils IF (Déterministes)               │       │
│  │  lookup_entity | get_neighbors | find_common... │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Elasticsearch │
                    │    (SIREN)    │
                    └───────────────┘
```

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou: .venv\Scripts\activate  # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer l'environnement
cp .env.example .env
# Éditer .env avec votre clé OpenAI
```

## Configuration

Créez un fichier `.env` à partir de `.env.example`:

```env
# Obligatoire
OPENAI_API_KEY=sk-votre-cle-api

# Optionnel (avec valeurs par défaut)
OPENAI_MODEL=gpt-4o-mini
ES_URL=https://localhost:9220
ES_USER=sirenadmin
ES_PASSWORD=password
ES_VERIFY_SSL=false
```

## Utilisation

### Mode interactif

```bash
python main.py
```

Puis posez vos questions en langage naturel:
```
🔍 Votre question: Trouve un lien entre MongoDB et Union Square Ventures
```

### Mode requête unique

```bash
python main.py "Quels sont les investisseurs de Uber?"
```

## Exemples de requêtes

- "Trouve un lien entre l'entreprise A et l'entreprise B"
- "Quels investisseurs ont investi dans Airbnb entre 2010 et 2015?"
- "Trouve les entreprises qui ont des investisseurs communs avec Tesla"
- "Liste les investissements de plus de 10 millions USD en 2012"

## Structure du projet

```
ZoneDeTravailDeClaude/
├── main.py                 # Point d'entrée CLI
├── requirements.txt        # Dépendances Python
├── .env.example           # Template de configuration
├── agents/
│   ├── planner.py         # Agent Planificateur
│   ├── prioritizer.py     # Agent Priorisation
│   └── summarizer.py      # Agent Résumé
├── tools/
│   └── elasticsearch_tools.py  # Outils IF (foraging)
└── core/
    └── orchestrator.py    # Orchestrateur LangGraph
```

## Les 3 Agents

### 1. Planificateur (PlannerAgent)
Reçoit la requête utilisateur et crée un plan d'action structuré.
- Analyse la question
- Identifie les entités
- Décompose en étapes
- Spécifie les outils à utiliser

### 2. Priorisation (PrioritizerAgent)
Optimise l'exploration du graphe.
- Analyse les résultats partiels
- Évalue les nœuds candidats
- Priorise les pistes prometteuses
- Évite l'exploration inutile

### 3. Résumé (SummarizerAgent)
Transforme les résultats bruts en réponse lisible.
- Synthétise les découvertes
- Structure l'information
- Produit une réponse claire

## Les Outils IF (Foraging)

Outils déterministes (sans LLM) pour interagir avec Elasticsearch:

| Outil | Description |
|-------|-------------|
| `lookup_entity` | Recherche une entité par nom |
| `get_neighbors` | Récupère les voisins d'un nœud |
| `find_common_investors` | Trouve les investisseurs communs |
| `find_investments_in_period` | Filtre les investissements par période |

## Dépendances principales

- **LangGraph**: Orchestration du workflow multi-agents
- **LangChain**: Framework pour les agents LLM
- **OpenAI**: LLM pour les agents intelligents
- **Elasticsearch**: Connexion à SIREN Investigate
