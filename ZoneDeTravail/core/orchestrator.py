"""
Orchestrateur LangGraph - Coordonne les agents et les outils pour répondre aux requêtes.
C'est le coeur du système multi-agents.
"""

import json
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import openai  # Pour capturer BadRequestError

from agents.planner import PlannerAgent
from agents.prioritizer import PrioritizerAgent
from agents.summarizer import SummarizerAgent
from tools.elasticsearch_tools import TOOLS, get_es_tools


class GraphState(TypedDict):
    """État du graphe partagé entre tous les noeuds."""
    # Requête initiale
    query: str
    # Plan d'action généré
    plan: str
    # Résultats d'exploration accumulés
    exploration_results: list[dict]
    # Noeuds explorés (pour éviter les boucles)
    explored_nodes: list[dict]
    # Noeuds candidats à explorer
    candidate_nodes: list[dict]
    # Compteur d'itérations (pour éviter boucles infinies)
    iteration: int
    # Résumé final
    summary: str
    # Messages pour le LLM exécuteur
    messages: Annotated[list, add_messages]
    # Flag de fin
    finished: bool
    # Flag si le contexte a été dépassé (résultats partiels)
    context_exceeded: bool


class GraphOrchestrator:
    """
    Orchestrateur principal utilisant LangGraph.
    Coordonne le Planificateur, l'Exécuteur (avec outils), le Priorisation et le Summarizer.
    """

    MAX_ITERATIONS = 15  # Limite de sécurité (augmentée pour plus d'exploration)
    CONDENSE_THRESHOLD = 25  # Seuil pour déclencher la condensation du contexte
    KEEP_RECENT_MESSAGES = 6  # Messages récents à garder après condensation

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

        # Initialiser les agents
        self.planner = PlannerAgent(model_name=model_name)
        self.prioritizer = PrioritizerAgent(model_name=model_name)
        self.summarizer = SummarizerAgent(model_name=model_name)

        # LLM exécuteur avec outils
        self.executor_llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.executor_llm_with_tools = self.executor_llm.bind_tools(TOOLS)

        # Construire le graphe
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construit le graphe LangGraph."""

        # Créer le graphe
        workflow = StateGraph(GraphState)

        # Ajouter les noeuds
        workflow.add_node("planner", self._plan_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("tools", ToolNode(TOOLS))
        workflow.add_node("prioritizer", self._prioritizer_node)
        workflow.add_node("summarizer", self._summarizer_node)

        # Définir le point d'entrée
        workflow.set_entry_point("planner")

        # Ajouter les transitions
        workflow.add_edge("planner", "executor")

        # Après executor: soit appeler tools, soit passer au prioritizer/summarizer
        workflow.add_conditional_edges(
            "executor",
            self._should_use_tools,
            {
                "tools": "tools",
                "prioritizer": "prioritizer",
                "summarizer": "summarizer"
            }
        )

        # Après tools: retourner à executor
        workflow.add_edge("tools", "executor")

        # Après prioritizer: retourner à executor ou terminer
        workflow.add_conditional_edges(
            "prioritizer",
            self._should_continue_exploration,
            {
                "executor": "executor",
                "summarizer": "summarizer"
            }
        )

        # Summarizer est la fin
        workflow.add_edge("summarizer", END)

        return workflow.compile()

    def _plan_node(self, state: GraphState) -> dict:
        """Noeud Planificateur - Crée le plan d'action."""
        print("\n[Planificateur] Création du plan d'action...")

        plan = self.planner.create_plan(state["query"])
        print(f"Plan créé:\n{plan}\n")

        # Préparer le message système pour l'exécuteur
        system_message = SystemMessage(content=f"""Tu es un agent exécuteur qui suit un plan d'action pour explorer un graphe d'entreprises.

PLAN À SUIVRE:
{plan}

INSTRUCTIONS:
1. Exécute le plan étape par étape en utilisant les outils disponibles
2. Après chaque outil, analyse le résultat
3. Adapte le plan si nécessaire (ex: si une entité n'est pas trouvée)
4. Quand le plan est complété, dis "EXPLORATION_COMPLETE"

Commence par la première étape du plan.""")

        return {
            "plan": plan,
            "messages": [system_message],
            "iteration": 0,
            "exploration_results": [],
            "explored_nodes": [],
            "candidate_nodes": [],
            "finished": False
        }

    def _condense_context(self, messages: list, query: str) -> list:
        """
        Condense le contexte en résumant les découvertes intermédiaires.

        Au lieu de tronquer (perdre l'info), on RÉSUME (préserver l'info condensée).
        C'est la vraie Option 2 : optimisation intelligente du contexte.
        """
        print("   Condensation du contexte en cours...")

        # Séparer le message système
        system_msg = None
        other_messages = messages
        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            other_messages = messages[1:]

        # Identifier les "unités" complètes (préserver paires tool_calls/tool)
        units = []
        i = len(other_messages) - 1
        while i >= 0:
            msg = other_messages[i]
            if isinstance(msg, ToolMessage):
                unit = [msg]
                i -= 1
                while i >= 0 and isinstance(other_messages[i], ToolMessage):
                    unit.insert(0, other_messages[i])
                    i -= 1
                if i >= 0 and isinstance(other_messages[i], AIMessage):
                    ai_msg = other_messages[i]
                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        unit.insert(0, ai_msg)
                        i -= 1
                units.insert(0, unit)
            else:
                units.insert(0, [msg])
                i -= 1

        # Garder les 2-3 dernières unités complètes pour continuité
        recent_units = units[-3:] if len(units) > 3 else units
        recent_messages = [msg for unit in recent_units for msg in unit]

        # Messages à résumer = tout sauf les récents
        units_to_summarize = units[:-3] if len(units) > 3 else []
        messages_to_summarize = [msg for unit in units_to_summarize for msg in unit]

        if not messages_to_summarize:
            return messages  # Rien à condenser

        # Extraire le contenu à résumer
        content_to_summarize = []
        for msg in messages_to_summarize:
            if isinstance(msg, AIMessage) and msg.content:
                content_to_summarize.append(f"[Assistant]: {msg.content[:500]}")
            elif isinstance(msg, ToolMessage):
                # Résumer les résultats d'outils (souvent volumineux)
                content = str(msg.content)[:300]
                content_to_summarize.append(f"[Outil]: {content}")
            elif isinstance(msg, HumanMessage):
                content_to_summarize.append(f"[Contexte]: {msg.content[:200]}")

        # Demander au LLM de résumer
        summary_prompt = f"""Résume de manière CONCISE les découvertes faites jusqu'ici pour répondre à: "{query}"

Informations à résumer:
{chr(10).join(content_to_summarize[-15:])}

Instructions:
- Liste les entités trouvées (companies, investors) avec leurs IDs
- Liste les faits clés découverts (montants, dates, relations)
- Sois TRÈS concis (max 300 mots)
- Format: bullet points"""

        try:
            summary_response = self.executor_llm.invoke([HumanMessage(content=summary_prompt)])
            summary_content = summary_response.content
        except Exception as e:
            # En cas d'erreur, faire un résumé basique
            summary_content = "Résumé non disponible - exploration en cours."

        # Créer le message de résumé condensé
        condensed_message = HumanMessage(content=f"""RÉSUMÉ DES DÉCOUVERTES PRÉCÉDENTES:
{summary_content}

Continue l'exploration à partir de ces informations.""")

        # Reconstruire la liste de messages
        result = []
        if system_msg:
            result.append(system_msg)
        result.append(condensed_message)
        result.extend(recent_messages)

        print(f"   Contexte condensé: {len(messages)} → {len(result)} messages")
        return result

    def _truncate_messages_safely(self, messages: list, max_messages: int) -> list:
        """
        Tronque les messages en préservant les paires tool_calls/tool_response.

        OpenAI exige que les ToolMessage suivent immédiatement le AIMessage
        avec tool_calls correspondant. Cette méthode garantit qu'on ne coupe
        jamais au milieu d'une telle paire.
        """
        if len(messages) <= max_messages:
            return messages

        # Séparer le message système (à toujours garder)
        system_msg = None
        other_messages = messages
        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            other_messages = messages[1:]
            max_messages -= 1  # Réserver une place pour le système

        # Identifier les "unités" de messages (groupes tool_calls + réponses)
        # On parcourt de la fin vers le début pour garder les plus récents
        units = []
        i = len(other_messages) - 1

        while i >= 0:
            msg = other_messages[i]

            if isinstance(msg, ToolMessage):
                # C'est une réponse d'outil - trouver le AIMessage avec tool_calls
                unit = [msg]
                i -= 1

                # Collecter toutes les ToolMessage consécutives
                while i >= 0 and isinstance(other_messages[i], ToolMessage):
                    unit.insert(0, other_messages[i])
                    i -= 1

                # Le message précédent doit être le AIMessage avec tool_calls
                if i >= 0 and isinstance(other_messages[i], AIMessage):
                    ai_msg = other_messages[i]
                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        unit.insert(0, ai_msg)
                        i -= 1

                units.insert(0, unit)
            else:
                # Message standalone (HumanMessage, AIMessage sans tool_calls)
                units.insert(0, [msg])
                i -= 1

        # Sélectionner les unités les plus récentes qui tiennent dans la limite
        selected_messages = []
        for unit in reversed(units):
            if len(selected_messages) + len(unit) <= max_messages:
                selected_messages = unit + selected_messages
            else:
                break

        # Reconstruire la liste avec le message système
        if system_msg:
            return [system_msg] + selected_messages
        return selected_messages

    def _executor_node(self, state: GraphState) -> dict:
        """Noeud Exécuteur - Exécute le plan en utilisant les outils."""
        print(f"\n[Exécuteur] Itération {state['iteration'] + 1}...")

        # OPTION 2 (vraie): Condensation intelligente du contexte
        # Au lieu de tronquer (perdre info), on RÉSUME (préserver info condensée)
        messages = state["messages"]
        if len(messages) > self.CONDENSE_THRESHOLD:
            messages = self._condense_context(messages, state["query"])

        try:
            # Appeler le LLM avec les outils (messages optimisés)
            response = self.executor_llm_with_tools.invoke(messages)

            return {
                "messages": [response],
                "iteration": state["iteration"] + 1
            }

        except openai.BadRequestError as e:
            # Gérer le dépassement de contexte
            if "context_length_exceeded" in str(e):
                print("\n[Exécuteur] Contexte trop grand - passage au résumé avec les résultats partiels...")

                # Créer un message qui forcera le passage au summarizer
                error_message = AIMessage(content="""EXPLORATION_COMPLETE

Note: L'exploration a été interrompue car le contexte était trop volumineux.
Les résultats ci-dessous sont partiels mais contiennent les informations trouvées jusqu'à présent.""")

                return {
                    "messages": [error_message],
                    "iteration": state["iteration"] + 1,
                    "context_exceeded": True
                }
            else:
                # Autre erreur OpenAI - la propager
                raise

    def _should_use_tools(self, state: GraphState) -> str:
        """Décide si on doit appeler des outils ou passer à la suite."""
        last_message = state["messages"][-1]

        # Vérifier si le LLM veut utiliser des outils
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("   → Appel d'outils...")
            return "tools"

        # Vérifier si l'exploration est terminée
        if "EXPLORATION_COMPLETE" in str(last_message.content):
            print("   → Exploration terminée, passage au résumé...")
            return "summarizer"

        # Vérifier la limite d'itérations
        if state["iteration"] >= self.MAX_ITERATIONS:
            print("   → Limite d'itérations atteinte, passage au résumé...")
            return "summarizer"

        # Sinon, passer au prioritizer pour décider de la suite
        print("   → Passage au priorisation...")
        return "prioritizer"

    def _prioritizer_node(self, state: GraphState) -> dict:
        """Noeud Priorisation - Décide des prochains noeuds à explorer."""
        print("\n[Priorisation] Analyse des prochaines étapes...")

        # Extraire les résultats actuels des messages
        results_summary = self._extract_results_summary(state["messages"])

        priorities = self.prioritizer.prioritize(
            objective=state["query"],
            explored_nodes=state["explored_nodes"],
            candidate_nodes=state["candidate_nodes"],
            partial_results=results_summary
        )

        print(f"Priorités:\n{priorities}\n")

        # Ajouter les priorités comme message pour l'exécuteur
        priority_message = HumanMessage(content=f"""Voici les noeuds prioritaires à explorer:

{priorities}

Continue l'exploration ou dis "EXPLORATION_COMPLETE" si tu as assez d'informations.""")

        return {"messages": [priority_message]}

    def _should_continue_exploration(self, state: GraphState) -> str:
        """Décide si on continue l'exploration ou si on résume."""
        # Simple heuristique: si on a dépassé un certain nombre d'itérations
        if state["iteration"] >= self.MAX_ITERATIONS // 2:
            return "summarizer"
        return "executor"

    def _summarizer_node(self, state: GraphState) -> dict:
        """Noeud Summarizer - Produit le résumé final."""
        print("\n[Summarizer] Production du résumé final...")

        # Extraire tous les résultats des messages
        raw_results = self._extract_results_summary(state["messages"])

        # Ajouter une note si le contexte a été dépassé
        if state.get("context_exceeded", False):
            raw_results += "\n\nNOTE: L'exploration a été interrompue (contexte trop volumineux). Ces résultats sont partiels."

        summary = self.summarizer.summarize(
            original_query=state["query"],
            plan=state["plan"],
            raw_results=raw_results
        )

        # Ajouter un avertissement en début de résumé si nécessaire
        if state.get("context_exceeded", False):
            summary = "**Résultats partiels** (exploration interrompue)\n\n" + summary

        return {"summary": summary, "finished": True}

    def _extract_results_summary(self, messages: list) -> str:
        """Extrait un résumé des résultats à partir des messages."""
        import json
        results = []

        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                content = str(msg.content)

                # Inclure les ToolMessages (contiennent les vraies données!)
                if isinstance(msg, ToolMessage):
                    # Essayer de parser le JSON pour extraire intelligemment
                    try:
                        data = json.loads(content) if content.startswith('{') or content.startswith('[') else None
                        if data:
                            summary = self._summarize_tool_data(data)
                            results.append(f"[Outil]: {summary}")
                        else:
                            results.append(f"[Outil]: {content[:1500]}")
                    except json.JSONDecodeError:
                        results.append(f"[Outil]: {content[:1500]}")

                # Messages de l'assistant (analyses, conclusions)
                elif isinstance(msg, AIMessage):
                    if len(content) > 50:
                        results.append(f"[Assistant]: {content[:1000]}")

                # Messages humains/contexte
                elif isinstance(msg, HumanMessage):
                    if len(content) > 50 and "RÉSUMÉ DES DÉCOUVERTES" not in content:
                        results.append(f"[Contexte]: {content[:500]}")

        # Garder plus de messages pour avoir un meilleur contexte
        return "\n---\n".join(results[-20:])

    def _summarize_tool_data(self, data: dict | list) -> str:
        """Résume intelligemment les données d'un outil."""
        if isinstance(data, list):
            # Liste d'entités - extraire les labels
            labels = [item.get('label', item.get('id', '?')) for item in data[:50]]
            if len(data) > 50:
                return f"{len(data)} résultats: {', '.join(labels[:20])}... (et {len(data)-20} autres)"
            return f"{len(data)} résultats: {', '.join(labels)}"

        if isinstance(data, dict):
            parts = []

            # Cas lookup_entity: {total, items}
            if 'items' in data and 'total' in data:
                items = data['items']
                labels = [item.get('label', '?') for item in items]
                parts.append(f"Trouvé {data['total']} entités: {', '.join(labels)}")

            # Cas get_neighbors: {investments, investors} ou {investments, companies}
            if 'companies' in data:
                companies = data['companies']
                labels = [c.get('label', '?') for c in companies]
                parts.append(f"{len(companies)} companies: {', '.join(labels)}")

            if 'investors' in data:
                investors = data['investors']
                labels = [i.get('label', '?') for i in investors]
                parts.append(f"{len(investors)} investors: {', '.join(labels)}")

            if 'investments' in data:
                investments = data['investments']
                # Extraire infos clés des investissements
                inv_summaries = []
                for inv in investments[:10]:
                    amount = inv.get('raised_amount', 0)
                    year = inv.get('funded_year', '?')
                    currency = inv.get('raised_currency_code', '')
                    inv_summaries.append(f"{year}: {amount:,.0f} {currency}" if amount else f"{year}")
                parts.append(f"{len(investments)} investments: {', '.join(inv_summaries)}")
                if len(investments) > 10:
                    parts[-1] += f"... (et {len(investments)-10} autres)"

            if parts:
                return " | ".join(parts)

        # Fallback: représentation JSON tronquée
        import json
        return json.dumps(data, ensure_ascii=False)[:1500]

    def run(self, query: str) -> str:
        """
        Exécute une requête d'investigation.

        Args:
            query: La question de l'utilisateur en langage naturel

        Returns:
            Le résumé des résultats
        """
        print(f"\n{'='*60}")
        print(f"Nouvelle investigation: {query}")
        print(f"{'='*60}")

        # État initial
        initial_state = {
            "query": query,
            "plan": "",
            "exploration_results": [],
            "explored_nodes": [],
            "candidate_nodes": [],
            "iteration": 0,
            "summary": "",
            "messages": [],
            "finished": False,
            "context_exceeded": False
        }

        # Exécuter le graphe
        final_state = self.graph.invoke(initial_state)

        return final_state.get("summary", "Aucun résultat trouvé.")
