"""
Agent Planificateur - Rédige un plan d'action pour répondre à la requête utilisateur.
Décompose la tâche en étapes et identifie les outils à utiliser.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PLANNER_SYSTEM_PROMPT = """Tu es un agent planificateur spécialisé dans l'investigation de graphes d'entreprises.

Tu reçois une requête utilisateur et tu dois créer un PLAN D'ACTION clair et structuré.

## Outils disponibles:
- lookup_entity(index, label): Recherche une entité par son nom. index = 'company', 'investor', ou 'investment'
- get_neighbors(entity_type, entity_id): Récupère les voisins d'un noeud (investments, investors, companies liés)
- find_common_investors(company_id_a, company_id_b): Trouve les investisseurs communs entre 2 entreprises
- find_investments_in_period(year_min, year_max, currency_code, min_amount): Filtre les investissements

## Structure des données:
- company: id, label, city, countrycode
- investor: id, label
- investment: id, label, funded_year, raised_amount, raised_currency_code, companies[], investors[]

## Ton rôle:
1. Analyser la requête utilisateur
2. Identifier les entités mentionnées
3. Décomposer en étapes logiques
4. Spécifier quels outils utiliser à chaque étape

## Format de sortie:
Retourne un plan structuré avec des étapes numérotées, chaque étape indiquant:
- L'objectif de l'étape
- L'outil à utiliser
- Les paramètres attendus

Sois concis et précis. Ne fais pas d'hypothèses sur les données - le plan sera exécuté ensuite."""


class PlannerAgent:
    """Agent qui crée un plan d'action pour l'investigation."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", "Requête utilisateur: {query}\n\nCrée un plan d'action.")
        ])
        self.chain = self.prompt | self.llm

    def create_plan(self, query: str) -> str:
        """
        Crée un plan d'action pour la requête donnée.

        Args:
            query: La requête utilisateur en langage naturel

        Returns:
            Le plan d'action sous forme de texte structuré
        """
        response = self.chain.invoke({"query": query})
        return response.content
