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

    # OPTION 2: Champs essentiels à garder pour chaque type d'entité
    ESSENTIAL_FIELDS = {
        "company": ["id", "label", "city", "countrycode", "founded_year", "homepage_url"],
        "investor": ["id", "label"],
        "investment": ["id", "label", "funded_year", "raised_amount", "raised_currency_code", "companies", "investors"]
    }

    # Nombre max d'items retournés par requête
    # Augmenté pour ne pas perdre d'information (vraie Option 2)
    MAX_ITEMS = 50

    def __init__(self):
        self.base_url = os.getenv("ES_URL", "https://localhost:9220")
        self.auth = (
            os.getenv("ES_USER", "sirenadmin"),
            os.getenv("ES_PASSWORD", "password")
        )
        self.verify_ssl = os.getenv("ES_VERIFY_SSL", "false").lower() == "true"

    def _clean_entity(self, entity: dict, entity_type: str) -> dict:
        """
        OPTION 2: Nettoie une entité en ne gardant que les champs essentiels.
        Réduit significativement la taille des données retournées.
        """
        essential = self.ESSENTIAL_FIELDS.get(entity_type, ["id", "label"])
        return {k: v for k, v in entity.items() if k in essential and v is not None}

    def _search(self, index: str, query: dict, size: int = 10) -> dict:
        """Exécute une recherche Elasticsearch."""
        url = f"{self.base_url}/{index}/_search"
        # OPTION 2: Limiter le nombre d'items
        query["size"] = min(size, self.MAX_ITEMS)
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

        # OPTION 2: Nettoyer les entités pour réduire la taille
        items = [self._clean_entity(hit["_source"], index) for hit in hits]

        return {
            "total": total,
            "items": items
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
            # Trouver les investments liés à cette company (limité par MAX_ITEMS)
            inv_query = {"query": {"terms": {"companies": [entity_id]}}}
            inv_res = self._search("investment", inv_query, self.MAX_ITEMS)
            investments_raw = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            # OPTION 2: Nettoyer les résultats
            neighbors["investments"] = [self._clean_entity(inv, "investment") for inv in investments_raw]

            # Extraire les investors de ces investments
            investor_ids = set()
            for inv in investments_raw:
                inv_list = inv.get("investors") or []
                investor_ids.update(inv_list)

            if investor_ids:
                inv_query = {"query": {"terms": {"id": list(investor_ids)[:self.MAX_ITEMS]}}}
                inv_res = self._search("investor", inv_query, self.MAX_ITEMS)
                neighbors["investors"] = [self._clean_entity(hit["_source"], "investor") for hit in inv_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["investors"] = []

        elif entity_type == "investor":
            # Trouver les investments de cet investor (limité par MAX_ITEMS)
            inv_query = {"query": {"terms": {"investors": [entity_id]}}}
            inv_res = self._search("investment", inv_query, self.MAX_ITEMS)
            investments_raw = [hit["_source"] for hit in inv_res.get("hits", {}).get("hits", [])]
            # OPTION 2: Nettoyer les résultats
            neighbors["investments"] = [self._clean_entity(inv, "investment") for inv in investments_raw]

            # Extraire les companies de ces investments
            company_ids = set()
            for inv in investments_raw:
                comp_list = inv.get("companies") or []
                company_ids.update(comp_list)

            if company_ids:
                comp_query = {"query": {"terms": {"id": list(company_ids)[:self.MAX_ITEMS]}}}
                comp_res = self._search("company", comp_query, self.MAX_ITEMS)
                neighbors["companies"] = [self._clean_entity(hit["_source"], "company") for hit in comp_res.get("hits", {}).get("hits", [])]
            else:
                neighbors["companies"] = []

        elif entity_type == "investment":
            # Récupérer l'investment
            entity = self.get_entity_by_id("investment", entity_id)
            if entity:
                # Récupérer les companies
                company_ids = entity.get("companies") or []
                if company_ids:
                    comp_query = {"query": {"terms": {"id": company_ids[:self.MAX_ITEMS]}}}
                    comp_res = self._search("company", comp_query, self.MAX_ITEMS)
                    neighbors["companies"] = [self._clean_entity(hit["_source"], "company") for hit in comp_res.get("hits", {}).get("hits", [])]
                else:
                    neighbors["companies"] = []

                # Récupérer les investors
                investor_ids = entity.get("investors") or []
                if investor_ids:
                    inv_query = {"query": {"terms": {"id": investor_ids[:self.MAX_ITEMS]}}}
                    inv_res = self._search("investor", inv_query, self.MAX_ITEMS)
                    neighbors["investors"] = [self._clean_entity(hit["_source"], "investor") for hit in inv_res.get("hits", {}).get("hits", [])]
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

        # Récupérer les détails (limité par MAX_ITEMS)
        query = {"query": {"terms": {"id": list(common_ids)[:self.MAX_ITEMS]}}}
        res = self._search("investor", query, self.MAX_ITEMS)

        # OPTION 2: Nettoyer les résultats
        return [self._clean_entity(hit["_source"], "investor") for hit in res.get("hits", {}).get("hits", [])]

    def find_investments_in_period(
        self,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        currency_code: Optional[str] = None,
        min_amount: Optional[float] = None,
        company_id: Optional[str] = None,
        investor_id: Optional[str] = None,
        size: int = 50
    ) -> list[dict[str, Any]]:
        """
        Recherche des investissements selon des critères temporels, financiers et par entité.

        Args:
            year_min: Année minimum
            year_max: Année maximum
            currency_code: Code devise (ex: 'USD', 'EUR')
            min_amount: Montant minimum
            company_id: ID d'une company pour filtrer ses investissements reçus
            investor_id: ID d'un investor pour filtrer ses investissements faits
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

        # Filtrer par company (investissements REÇUS par cette company)
        if company_id:
            filters.append({"terms": {"companies": [company_id]}})

        # Filtrer par investor (investissements FAITS par cet investor)
        if investor_id:
            filters.append({"terms": {"investors": [investor_id]}})

        query = {
            "query": {"bool": {"filter": filters}} if filters else {"match_all": {}}
        }

        # OPTION 2: Limiter et nettoyer les résultats
        res = self._search("investment", query, min(size, self.MAX_ITEMS))
        return [self._clean_entity(hit["_source"], "investment") for hit in res.get("hits", {}).get("hits", [])]


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

    ⚠️ RÈGLE CRITIQUE: entity_type = type de l'entité dont vous avez l'ID !

    ┌─────────────────────────────────────────────────────────────────────┐
    │ SI VOUS CHERCHEZ...          │ UTILISEZ entity_type = ...          │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Les investisseurs d'Airbnb   │ "company"  (car Airbnb est company) │
    │ Les companies de Jeff Bezos  │ "investor" (car Bezos est investor) │
    │ Les détails d'un funding     │ "investment" (car c'est investment) │
    └─────────────────────────────────────────────────────────────────────┘

    ✅ CORRECT: get_neighbors("company", "company/amazon")
       → Retourne les investors ET investments d'Amazon

    ✅ CORRECT: get_neighbors("investor", "financial-organization/investor/kleiner-perkins-caufield-byers")
       → Retourne les companies ET investments de Kleiner Perkins

    ❌ FAUX: get_neighbors("investor", "company/amazon")
       → Ne fonctionnera pas ! Amazon n'est pas un investor !

    Retours selon entity_type:
    - "company"    → {investments: [...], investors: [...]}
    - "investor"   → {investments: [...], companies: [...]}
    - "investment" → {companies: [...], investors: [...]}

    Args:
        entity_type: "company", "investor", ou "investment" - DOIT correspondre au type de l'entity_id !
        entity_id: L'ID complet de l'entité (ex: "company/airbnb", "person/investor/jeff-bezos")

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
    company_id: str = None,
    investor_id: str = None,
    size: int = 50
) -> list:
    """
    Recherche des investissements selon des critères temporels, financiers et par entité.

    ⚠️ IMPORTANT: Sans company_id ou investor_id, retourne TOUS les investissements de la période !

    Cas d'utilisation:
    - Investissements REÇUS par Facebook: company_id="company/facebook"
    - Investissements FAITS par Kleiner Perkins: investor_id="financial-organization/investor/kleiner-perkins-caufield-byers"
    - Tous les investissements de 2010: year_min=2010, year_max=2010 (sans filtrer par entité)

    Args:
        year_min: Année minimum (ex: 2010)
        year_max: Année maximum (ex: 2020)
        currency_code: Code devise - 'USD', 'EUR', etc.
        min_amount: Montant minimum d'investissement
        company_id: ID d'une company pour voir ses investissements REÇUS
        investor_id: ID d'un investor pour voir ses investissements FAITS
        size: Nombre maximum de résultats

    Returns:
        Liste des investissements correspondant aux critères
    """
    return get_es_tools().find_investments_in_period(
        year_min, year_max, currency_code, min_amount, company_id, investor_id, size
    )


# Liste de tous les outils disponibles pour LangGraph
TOOLS = [lookup_entity, get_neighbors, find_common_investors, find_investments_in_period]
