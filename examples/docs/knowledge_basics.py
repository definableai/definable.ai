from definable.knowledge import Document, Knowledge

from support import MockVectorDB


knowledge = Knowledge(vector_db=MockVectorDB(), context_format="markdown")
document_ids = knowledge.add(
  Document(content="Definable uses knowledge to ground answers in retrieved context."),
  chunk=False,
)
context = knowledge.format_context(knowledge.vector_db.search("ground", limit=1))

assert len(document_ids) == 1
assert "Definable uses knowledge" in context
