"""
Agent Priorisation - Détermine les prochains noeuds à explorer dans le graphe.
Aide à optimiser l'exploration en priorisant les pistes les plus prometteuses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PRIORITIZER_SYSTEM_PROMPT = """Tu es un agent de priorisation spécialisé dans l'exploration de graphes d'investigation.

## Ton rôle:
Analyser les résultats d'exploration actuels et déterminer quels noeuds explorer en priorité.

## Contexte:
Tu reçois:
- L'objectif de l'investigation
- Les noeuds déjà explorés
- Les noeuds candidats (voisins non encore explorés)
- Les résultats partiels obtenus

## Critères de priorisation:
1. **Pertinence**: Le noeud semble-t-il lié à l'objectif ?
2. **Connectivité**: Le noeud a-t-il beaucoup de connexions potentielles ?
3. **Temporalité**: Le noeud correspond-il à la période recherchée ?
4. **Nouveauté**: Le noeud apporte-t-il de nouvelles informations ?

## Format de sortie:
Retourne une liste ordonnée des noeuds à explorer, avec pour chacun:
- L'identifiant du noeud
- Le type (company, investor, investment)
- La raison de la priorité (1 phrase)
- Un score de priorité (1-10)

Sois stratégique: mieux vaut explorer peu de noeuds pertinents que beaucoup de noeuds au hasard."""


class PrioritizerAgent:
    """Agent qui priorise les noeuds à explorer dans le graphe."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PRIORITIZER_SYSTEM_PROMPT),
            ("human", """Objectif: {objective}

Noeuds déjà explorés:
{explored_nodes}

Noeuds candidats (voisins non explorés):
{candidate_nodes}

Résultats partiels:
{partial_results}

Quels noeuds explorer en priorité ?""")
        ])
        self.chain = self.prompt | self.llm

    def prioritize(
        self,
        objective: str,
        explored_nodes: list[dict],
        candidate_nodes: list[dict],
        partial_results: str
    ) -> str:
        """
        Priorise les noeuds candidats pour l'exploration.

        Args:
            objective: L'objectif de l'investigation
            explored_nodes: Liste des noeuds déjà visités
            candidate_nodes: Liste des noeuds candidats à explorer
            partial_results: Résumé des résultats obtenus jusqu'ici

        Returns:
            Liste ordonnée des noeuds à explorer avec justification
        """
        # Formater les listes pour le prompt
        explored_str = "\n".join([
            f"- {n.get('type', '?')}: {n.get('label', n.get('id', '?'))}"
            for n in explored_nodes
        ]) if explored_nodes else "Aucun"

        candidates_str = "\n".join([
            f"- {n.get('type', '?')}: {n.get('label', n.get('id', '?'))}"
            for n in candidate_nodes
        ]) if candidate_nodes else "Aucun"

        response = self.chain.invoke({
            "objective": objective,
            "explored_nodes": explored_str,
            "candidate_nodes": candidates_str,
            "partial_results": partial_results or "Aucun résultat pour l'instant"
        })

        return response.content
