"""
Agent Résumé - Produit un résumé lisible des résultats d'investigation pour l'utilisateur.
Transforme les données brutes en insights compréhensibles.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SUMMARIZER_SYSTEM_PROMPT = """Tu es un agent de synthèse spécialisé dans la présentation de résultats d'investigation.

## Ton rôle:
Transformer les résultats bruts d'exploration de graphe en un résumé clair et actionnable pour l'utilisateur humain.

## Principes:
1. **Clarté**: Utilise un langage simple et direct
2. **Structure**: Organise l'information de manière logique
3. **Pertinence**: Mets en avant les découvertes importantes
4. **Contexte**: Rappelle brièvement la question initiale
5. **Honnêteté**: Si les résultats sont incomplets, dis-le clairement

## Format de sortie:
Structure ton résumé ainsi:

### Réponse à votre question
[Réponse directe et concise]

### Découvertes clés
[Points importants trouvés]

### Détails
[Informations supplémentaires pertinentes]

### Limites
[Ce qui n'a pas pu être trouvé ou vérifié, si applicable]

Sois concis mais complet. L'utilisateur doit comprendre les résultats sans avoir besoin de poser des questions de suivi."""


class SummarizerAgent:
    """Agent qui résume les résultats d'investigation pour l'utilisateur."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        # Température légèrement plus haute pour un style plus naturel
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_SYSTEM_PROMPT),
            ("human", """Question initiale de l'utilisateur:
{original_query}

Plan d'action exécuté:
{plan}

Résultats bruts de l'exploration:
{raw_results}

Produis un résumé clair et lisible pour l'utilisateur.""")
        ])
        self.chain = self.prompt | self.llm

    def summarize(self, original_query: str, plan: str, raw_results: str) -> str:
        """
        Produit un résumé lisible des résultats d'investigation.

        Args:
            original_query: La question initiale de l'utilisateur
            plan: Le plan d'action qui a été exécuté
            raw_results: Les résultats bruts de l'exploration

        Returns:
            Un résumé structuré et lisible
        """
        response = self.chain.invoke({
            "original_query": original_query,
            "plan": plan,
            "raw_results": raw_results
        })

        return response.content
