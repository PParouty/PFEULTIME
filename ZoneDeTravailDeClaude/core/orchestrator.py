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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

    MAX_ITERATIONS = 10  # Limite de sécurité

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
        print("\n📋 [Planificateur] Création du plan d'action...")

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

    def _executor_node(self, state: GraphState) -> dict:
        """Noeud Exécuteur - Exécute le plan en utilisant les outils."""
        print(f"\n🔧 [Exécuteur] Itération {state['iteration'] + 1}...")

        try:
            # Appeler le LLM avec les outils
            response = self.executor_llm_with_tools.invoke(state["messages"])

            return {
                "messages": [response],
                "iteration": state["iteration"] + 1
            }

        except openai.BadRequestError as e:
            # Gérer le dépassement de contexte
            if "context_length_exceeded" in str(e):
                print("\n⚠️  [Exécuteur] Contexte trop grand - passage au résumé avec les résultats partiels...")

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
        print("\n🎯 [Priorisation] Analyse des prochaines étapes...")

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
        print("\n📝 [Summarizer] Production du résumé final...")

        # Extraire tous les résultats des messages
        raw_results = self._extract_results_summary(state["messages"])

        # Ajouter une note si le contexte a été dépassé
        if state.get("context_exceeded", False):
            raw_results += "\n\n⚠️ NOTE: L'exploration a été interrompue (contexte trop volumineux). Ces résultats sont partiels."

        summary = self.summarizer.summarize(
            original_query=state["query"],
            plan=state["plan"],
            raw_results=raw_results
        )

        # Ajouter un avertissement en début de résumé si nécessaire
        if state.get("context_exceeded", False):
            summary = "⚠️ **Résultats partiels** (exploration interrompue)\n\n" + summary

        return {"summary": summary, "finished": True}

    def _extract_results_summary(self, messages: list) -> str:
        """Extrait un résumé des résultats à partir des messages."""
        results = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                # Filtrer les messages système
                if isinstance(msg, (AIMessage, HumanMessage)):
                    content = str(msg.content)
                    if len(content) > 50:  # Ignorer les messages très courts
                        results.append(content[:500])  # Limiter la taille

        return "\n---\n".join(results[-10:])  # Garder les 10 derniers

    def run(self, query: str) -> str:
        """
        Exécute une requête d'investigation.

        Args:
            query: La question de l'utilisateur en langage naturel

        Returns:
            Le résumé des résultats
        """
        print(f"\n{'='*60}")
        print(f"🔍 Nouvelle investigation: {query}")
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
