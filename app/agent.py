from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from loguru import logger

from app.config import get_settings
from app.tools import TOOLS

settings = get_settings()


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sos un asistente experto en Google Cloud Platform (GCP).
Respondé preguntas técnicas sobre GCP usando las herramientas disponibles.

INSTRUCCIONES:
1. Usá search_gcp_docs para buscar información antes de responder
2. Citá las fuentes con título y URL
3. Al final usá get_related_documentation para sugerir docs relacionados
4. Si no encontrás información, decilo claramente
5. No inventes información
"""


# ── STATE ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── AGENT ─────────────────────────────────────────────────────────────────────

class GCPAgent:
    def __init__(self):
        # LLM — Groq con Llama 3.1 70B
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
        ).bind_tools(TOOLS)

        # Memoria por thread_id
        self.memory = MemorySaver()

        # Construir el grafo
        self.graph = self._build_graph()

        logger.info("GCPAgent inicializado correctamente")

    def _build_graph(self) -> StateGraph:
        """Construye el grafo LangGraph del agente."""
        graph = StateGraph(AgentState)

        # Nodos
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", ToolNode(TOOLS))

        # Punto de entrada
        graph.set_entry_point("llm")

        # Aristas condicionales — si el LLM quiere usar una herramienta
        # va al nodo tools, si no va al END
        graph.add_conditional_edges(
            "llm",
            self._should_use_tool,
            {
                "tools": "tools",
                "end":   END,
            }
        )

        # Después de ejecutar una herramienta, volvé al LLM
        graph.add_edge("tools", "llm")

        return graph.compile(checkpointer=self.memory)

    def _call_llm(self, state: AgentState) -> AgentState:
        messages = state["messages"]

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = self.llm.invoke(messages)
        logger.debug(f"LLM tool_calls: {getattr(response, 'tool_calls', None)}")

        return {"messages": [response]}
    def _should_use_tool(self, state: AgentState) -> str:
        """Decide si el LLM quiere usar una herramienta o terminar."""
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.debug(f"LLM quiere usar: {[t['name'] for t in last_message.tool_calls]}")
            return "tools"

        return "end"

    def query(self, question: str, thread_id: str = "default") -> dict:
        """
        Procesa una pregunta y retorna la respuesta estructurada.

        Args:
            question  : pregunta del usuario sobre GCP
            thread_id : ID de sesión para memoria conversacional

        Returns:
            dict con answer, sources y related
        """
        config = {"configurable": {"thread_id": thread_id}}

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        # Extraemos el último mensaje del agente
        last_message = result["messages"][-1]
        answer = last_message.content

        # Parseamos fuentes y relacionados del texto de respuesta
        sources, related = self._parse_response(result["messages"])

        return {
            "answer":    answer,
            "sources":   sources,
            "related":   related,
            "thread_id": thread_id,
        }

    def _parse_response(self, messages: list) -> tuple:
        """
        Extrae fuentes y docs relacionados de los mensajes del agente.
        Busca en los tool results los chunks recuperados.
        """
        sources = []
        related = []
        seen_urls = set()

        for msg in messages:
            # Tool results contienen los fragmentos recuperados
            if hasattr(msg, "name"):
                if msg.name == "search_gcp_docs" and msg.content:
                    # Parseamos los fragmentos del resultado
                    for block in msg.content.split("---"):
                        lines = block.strip().split("\n")
                        source = {}
                        for line in lines:
                            if line.startswith("Título:"):
                                source["title"] = line.replace("Título:", "").strip()
                            elif line.startswith("URL:"):
                                source["url"] = line.replace("URL:", "").strip()
                            elif line.startswith("Página:"):
                                try:
                                    source["page"] = int(line.replace("Página:", "").strip())
                                except ValueError:
                                    source["page"] = 0
                            elif line.startswith("Score:"):
                                try:
                                    source["relevance_score"] = float(line.replace("Score:", "").strip())
                                except ValueError:
                                    source["relevance_score"] = 0.0

                        if source.get("url") and source["url"] not in seen_urls:
                            seen_urls.add(source["url"])
                            sources.append(source)

                elif msg.name == "get_related_documentation" and msg.content:
                    for line in msg.content.strip().split("\n"):
                        if line.startswith("- "):
                            title = line[2:].strip()
                            related.append({"title": title, "url": "", "reason": "Documentación relacionada"})
                        elif line.strip().startswith("http"):
                            if related:
                                related[-1]["url"] = line.strip()

        return sources, related


# Instancia global del agente
agent = GCPAgent()