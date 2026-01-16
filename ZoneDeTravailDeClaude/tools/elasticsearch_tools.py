"""
Outils IF (Investigation Foraging) - Fonctions déterministes pour explorer le graphe SIREN.
Ces outils n'utilisent pas de LLM, ils exécutent des requêtes Elasticsearch précises.
"""

import os
from typing import Any, Optional
from elasticsearch import Elasticsearch
from langchain_core.tools import tool


class ElasticsearchTools:
    """Gestionnaire de connexion Elasticsearch et outils de foraging."""

    def __init__(self):
        self.es = Elasticsearch(
            os.getenv("ES_URL", "https://localhost:9220"),
            basic_auth=(
                os.getenv("ES_USER", "sirenadmin"),
                os.getenv("ES_PASSWORD", "password")
            ),
            verify_certs=os.getenv("ES_VERIFY_SSL", "false").lower() == "true"
        )

    def lookup_entity(self, index: str, label: str, size: int = 10) -> dict[str, Any]:
        """
        Recherche une entité par son label dans un index donné.

        Args:
            index: 'company', 'investor', ou 'investment'
            label: Le nom/label à rechercher
            size: Nombre max de résultats

        Returns:
            Dict avec 'total' et 'items'
        """
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"label": label}},
                        {"wildcard": {"label": {"value": f"*{label.lower()}*"}}}
                    ]
                }
            },
            "size": size
        }

        res = self.es.search(index=index, body=query)
        hits = res.get("hits", {}).get("hits", [])
        total = res.get("hits", {}).get("total", {}).get("value", 0)

        return {
            "total": total,
            "items": [hit["_source"] for hit in hits]
        }

    def get_entity_by_id(self, index: str, entity_id: str) -> Optional[dict[str, Any]]:
        """
        Récupère une entité par son ID.

        Args:
            index: 'company', 'investor', ou 'investment'
            entity_id: L'ID de l'entité

        Returns:
            L'entité ou None si non trouvée
        """
        query = {
            "query": {"term": {"id": entity_id}},
            "size": 1
        }

        res = self.es.search(index=index, body=query)
        hits = res.get("hits", {}).get("hits", [])

        return hits[0]["_source"] if hits else None

    def get_neighbors(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """
        Récupère les voisins d'un noeud dans le graphe.

        Pour une company: retourne ses investments et investors
        Pour un investor: retourne ses investments et companies investies
        Pour un investment: retourne ses companies et investors

        Args:
            entity_type: 'company', 'investor', ou 'investment'
            entity_id: L'ID de l'entité

        Returns:
            Dict avec les voisins par type
        """
        neighbors = {}

        if entity_type == "company":
            # Trouver les investments liés à cette company
            inv_query = {"query": {"terms": {"companies": [entity_id]}}, "size": 100}
            inv_res = self.es.search(index="investment", body=inv_query)
            investments = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            neighbors["investments"] = investments

            # Extraire les investors de ces investments
            investor_ids = set()
            for inv in investments:
                investor_ids.update(inv.get("investors", []))

            if investor_ids:
                inv_query = {"query": {"terms": {"id": list(investor_ids)}}, "size": 100}
                inv_res = self.es.search(index="investor", body=inv_query)
                neighbors["investors"] = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["investors"] = []

        elif entity_type == "investor":
            # Trouver les investments de cet investor
            inv_query = {"query": {"terms": {"investors": [entity_id]}}, "size": 100}
            inv_res = self.es.search(index="investment", body=inv_query)
            investments = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            neighbors["investments"] = investments

            # Extraire les companies de ces investments
            company_ids = set()
            for inv in investments:
                company_ids.update(inv.get("companies", []))

            if company_ids:
                comp_query = {"query": {"terms": {"id": list(company_ids)}}, "size": 100}
                comp_res = self.es.search(index="company", body=comp_query)
                neighbors["companies"] = [hit["_source"] for hit in comp_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["companies"] = []

        elif entity_type == "investment":
            # Récupérer l'investment
            entity = self.get_entity_by_id("investment", entity_id)
            if entity:
                # Récupérer les companies
                company_ids = entity.get("companies", [])
                if company_ids:
                    comp_query = {"query": {"terms": {"id": company_ids}}, "size": 100}
                    comp_res = self.es.search(index="company", body=comp_query)
                    neighbors["companies"] = [hit["_source"] for hit in comp_res.get("hits", {}).get("hits", [])]
                else:
                    neighbors["companies"] = []

                # Récupérer les investors
                investor_ids = entity.get("investors", [])
                if investor_ids:
                    inv_query = {"query": {"terms": {"id": investor_ids}}, "size": 100}
                    inv_res = self.es.search(index="investor", body=inv_query)
                    neighbors["investors"] = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
                else:
                    neighbors["investors"] = []

        return neighbors

    def find_common_investors(self, company_id_a: str, company_id_b: str) -> list[dict[str, Any]]:
        """
        Trouve les investisseurs communs entre deux entreprises.

        Args:
            company_id_a: ID de la première entreprise
            company_id_b: ID de la deuxième entreprise

        Returns:
            Liste des investisseurs communs
        """
        # Investisseurs de A
        neighbors_a = self.get_neighbors("company", company_id_a)
        investors_a = {inv.get("id") for inv in neighbors_a.get("investors", [])}

        # Investisseurs de B
        neighbors_b = self.get_neighbors("company", company_id_b)
        investors_b = {inv.get("id") for inv in neighbors_b.get("investors", [])}

        # Intersection
        common_ids = investors_a.intersection(investors_b)

        if not common_ids:
            return []

        # Récupérer les détails
        query = {"query": {"terms": {"id": list(common_ids)}}, "size": 100}
        res = self.es.search(index="investor", body=query)

        return [hit["_source"] for hit in res.get("hits", {}).get("hits", [])]

    def find_investments_in_period(
        self,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        currency_code: Optional[str] = None,
        min_amount: Optional[float] = None,
        size: int = 50
    ) -> list[dict[str, Any]]:
        """
        Recherche des investissements selon des critères temporels et financiers.

        Args:
            year_min: Année minimum
            year_max: Année maximum
            currency_code: Code devise (ex: 'USD', 'EUR')
            min_amount: Montant minimum
            size: Nombre max de résultats

        Returns:
            Liste des investissements correspondants
        """
        filters = []

        if year_min is not None or year_max is not None:
            year_range = {}
            if year_min:
                year_range["gte"] = year_min
            if year_max:
                year_range["lte"] = year_max
            filters.append({"range": {"funded_year": year_range}})

        if currency_code:
            filters.append({"term": {"raised_currency_code": currency_code}})

        if min_amount:
            filters.append({"range": {"raised_amount": {"gte": min_amount}}})

        query = {
            "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
            "size": size
        }

        res = self.es.search(index="investment", body=query)
        return [hit["_source"] for hit in res.get("hits", {}).get("hits", [])]


# Création des outils LangChain à partir de la classe
_es_tools = None

def get_es_tools() -> ElasticsearchTools:
    """Singleton pour les outils Elasticsearch."""
    global _es_tools
    if _es_tools is None:
        _es_tools = ElasticsearchTools()
    return _es_tools


@tool
def lookup_entity(index: str, label: str, size: int = 10) -> dict:
    """
    Recherche une entité (company, investor, investment) par son nom/label.

    Args:
        index: Type d'entité - 'company', 'investor', ou 'investment'
        label: Le nom à rechercher
        size: Nombre maximum de résultats (défaut: 10)

    Returns:
        Dictionnaire avec 'total' (nombre de résultats) et 'items' (liste des entités)
    """
    return get_es_tools().lookup_entity(index, label, size)


@tool
def get_neighbors(entity_type: str, entity_id: str) -> dict:
    """
    Récupère tous les voisins d'un noeud dans le graphe d'investigation.

    Pour une company: retourne ses investments et investors liés
    Pour un investor: retourne ses investments et companies investies
    Pour un investment: retourne ses companies et investors associés

    Args:
        entity_type: Type de l'entité - 'company', 'investor', ou 'investment'
        entity_id: L'identifiant unique de l'entité

    Returns:
        Dictionnaire avec les voisins groupés par type
    """
    return get_es_tools().get_neighbors(entity_type, entity_id)


@tool
def find_common_investors(company_id_a: str, company_id_b: str) -> list:
    """
    Trouve les investisseurs qui ont investi dans les deux entreprises.
    Utile pour trouver des liens entre deux entreprises.

    Args:
        company_id_a: ID de la première entreprise
        company_id_b: ID de la deuxième entreprise

    Returns:
        Liste des investisseurs communs avec leurs détails
    """
    return get_es_tools().find_common_investors(company_id_a, company_id_b)


@tool
def find_investments_in_period(
    year_min: int = None,
    year_max: int = None,
    currency_code: str = None,
    min_amount: float = None,
    size: int = 50
) -> list:
    """
    Recherche des investissements selon des critères temporels et financiers.

    Args:
        year_min: Année minimum (ex: 2000)
        year_max: Année maximum (ex: 2010)
        currency_code: Code devise - 'USD', 'EUR', etc.
        min_amount: Montant minimum d'investissement
        size: Nombre maximum de résultats

    Returns:
        Liste des investissements correspondant aux critères
    """
    return get_es_tools().find_investments_in_period(year_min, year_max, currency_code, min_amount, size)


# Liste de tous les outils disponibles pour LangGraph
TOOLS = [lookup_entity, get_neighbors, find_common_investors, find_investments_in_period]
