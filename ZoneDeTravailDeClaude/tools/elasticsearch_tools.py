"""
Outils IF (Investigation Foraging) - Fonctions déterministes pour explorer le graphe SIREN.
Ces outils n'utilisent pas de LLM, ils exécutent des requêtes Elasticsearch précises.
"""

import os
import warnings
import requests
from typing import Any, Optional
from langchain_core.tools import tool

# Désactiver les warnings SSL pour le développement
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")


class ElasticsearchTools:
    """Gestionnaire de connexion Elasticsearch et outils de foraging."""

    def __init__(self):
        self.base_url = os.getenv("ES_URL", "https://localhost:9220")
        self.auth = (
            os.getenv("ES_USER", "sirenadmin"),
            os.getenv("ES_PASSWORD", "password")
        )
        self.verify_ssl = os.getenv("ES_VERIFY_SSL", "false").lower() == "true"

    def _search(self, index: str, query: dict, size: int = 10) -> dict:
        """Exécute une recherche Elasticsearch."""
        url = f"{self.base_url}/{index}/_search"
        query["size"] = size
        response = requests.post(url, json=query, auth=self.auth, verify=self.verify_ssl)
        response.raise_for_status()
        return response.json()

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
            }
        }

        res = self._search(index, query, size)
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
        query = {"query": {"term": {"id": entity_id}}}
        res = self._search(index, query, 1)
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
            inv_query = {"query": {"terms": {"companies": [entity_id]}}}
            inv_res = self._search("investment", inv_query, 100)
            investments = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            neighbors["investments"] = investments

            # Extraire les investors de ces investments
            investor_ids = set()
            for inv in investments:
                inv_list = inv.get("investors") or []
                investor_ids.update(inv_list)

            if investor_ids:
                inv_query = {"query": {"terms": {"id": list(investor_ids)}}}
                inv_res = self._search("investor", inv_query, 100)
                neighbors["investors"] = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["investors"] = []

        elif entity_type == "investor":
            # Trouver les investments de cet investor
            inv_query = {"query": {"terms": {"investors": [entity_id]}}}
            inv_res = self._search("investment", inv_query, 100)
            investments = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            neighbors["investments"] = investments

            # Extraire les companies de ces investments
            company_ids = set()
            for inv in investments:
                comp_list = inv.get("companies") or []
                company_ids.update(comp_list)

            if company_ids:
                comp_query = {"query": {"terms": {"id": list(company_ids)}}}
                comp_res = self._search("company", comp_query, 100)
                neighbors["companies"] = [hit["_source"] for hit in comp_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["companies"] = []

        elif entity_type == "investment":
            # Récupérer l'investment
            entity = self.get_entity_by_id("investment", entity_id)
            if entity:
                # Récupérer les companies
                company_ids = entity.get("companies") or []
                if company_ids:
                    comp_query = {"query": {"terms": {"id": company_ids}}}
                    comp_res = self._search("company", comp_query, 100)
                    neighbors["companies"] = [hit["_source"] for hit in comp_res.get("hits", {}).get("hits", [])]
                else:
                    neighbors["companies"] = []

                # Récupérer les investors
                investor_ids = entity.get("investors") or []
                if investor_ids:
                    inv_query = {"query": {"terms": {"id": investor_ids}}}
                    inv_res = self._search("investor", inv_query, 100)
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
        query = {"query": {"terms": {"id": list(common_ids)}}}
        res = self._search("investor", query, 100)

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
            "query": {"bool": {"filter": filters}} if filters else {"match_all": {}}
        }

        res = self._search("investment", query, size)
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

    IMPORTANT: entity_type est le TYPE DE L'ENTITÉ SOURCE, pas le type des voisins recherchés !

    Exemples d'utilisation:
    - Pour trouver les investisseurs d'une COMPANY: get_neighbors("company", "company/airbnb")
    - Pour trouver les companies d'un INVESTOR: get_neighbors("investor", "person/investor/jeff-bezos")

    Ce que retourne chaque type:
    - entity_type="company" → retourne {investments: [...], investors: [...]}
    - entity_type="investor" → retourne {investments: [...], companies: [...]}
    - entity_type="investment" → retourne {companies: [...], investors: [...]}

    Args:
        entity_type: Type de l'entité SOURCE - 'company', 'investor', ou 'investment'
        entity_id: L'identifiant unique de l'entité (ex: 'company/airbnb')

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
