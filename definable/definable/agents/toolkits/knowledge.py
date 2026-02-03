"""Knowledge base toolkit for explicit agent access to RAG functionality."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from definable.agents.toolkit import Toolkit
from definable.tools.decorator import tool

if TYPE_CHECKING:
  from definable.knowledge import Knowledge
  from definable.tools.function import Function


class KnowledgeToolkit(Toolkit):
  """
  Toolkit providing explicit knowledge base search tools to agents.

  Unlike KnowledgeMiddleware (which automatically retrieves context),
  this toolkit gives agents direct control over when and how to search
  the knowledge base.

  Use cases:
    - Agent decides when retrieval is needed based on query
    - Multi-step retrieval with refined queries
    - Accessing specific documents by ID
    - Combining automatic and explicit search

  Example:
    from definable.agents import Agent
    from definable.agents.toolkits import KnowledgeToolkit
    from definable.knowledge import Knowledge

    kb = Knowledge(vector_db=..., embedder=...)
    kb.add(Document(content="Company policy: 20 days PTO."))

    agent = Agent(
      model=my_model,
      toolkits=[KnowledgeToolkit(knowledge=kb, top_k=5)],
      instructions="Search the knowledge base when you need information.",
    )

    response = agent.run("What is the PTO policy?")
    # Agent will call search_knowledge tool, then respond
  """

  def __init__(
    self,
    knowledge: "Knowledge",
    top_k: int = 5,
    rerank: bool = True,
  ):
    """
    Initialize the knowledge toolkit.

    Args:
      knowledge: Knowledge base instance to search.
      top_k: Default number of results to return.
      rerank: Whether to rerank search results.
    """
    super().__init__(dependencies={"_kb_toolkit_knowledge": knowledge})
    self._knowledge = knowledge
    self._top_k = top_k
    self._rerank = rerank

  @property
  def tools(self) -> List["Function"]:
    """Return the toolkit's tools."""
    return [self._search_knowledge, self._get_document_count]

  @property
  def _search_knowledge(self) -> "Function":
    """Create the search_knowledge tool."""

    @tool
    def search_knowledge(
      query: str,
      top_k: Optional[int] = None,
      dependencies: Optional[Dict[str, Any]] = None,
    ) -> str:
      """
      Search the knowledge base for relevant information.

      Use this tool when you need to find specific information from the
      knowledge base to answer user questions accurately.

      Args:
        query: The search query describing what information you need.
        top_k: Number of results to return (default: 5).

      Returns:
        Formatted text containing relevant documents with relevance scores.
      """
      kb = (dependencies or {}).get("_kb_toolkit_knowledge", self._knowledge)
      k = top_k if top_k is not None else self._top_k

      try:
        results = kb.search(query, top_k=k, rerank=self._rerank)
      except Exception as e:
        return f"Error searching knowledge base: {e}"

      if not results:
        return "No relevant documents found for the query."

      output_parts: List[str] = []
      for i, doc in enumerate(results):
        score_str = f" (relevance: {doc.reranking_score:.3f})" if doc.reranking_score else ""
        name_str = f" - {doc.name}" if doc.name else ""
        header = f"[{i + 1}]{name_str}{score_str}"
        output_parts.append(f"{header}\n{doc.content}")

      return "\n\n---\n\n".join(output_parts)

    return search_knowledge

  @property
  def _get_document_count(self) -> "Function":
    """Create the get_document_count tool."""

    @tool
    def get_document_count(
      dependencies: Optional[Dict[str, Any]] = None,
    ) -> str:
      """
      Get the total number of documents in the knowledge base.

      Use this to understand the size of the available knowledge.

      Returns:
        String indicating the document count.
      """
      kb = (dependencies or {}).get("_kb_toolkit_knowledge", self._knowledge)
      count = len(kb)
      return f"The knowledge base contains {count} document(s)."

    return get_document_count
